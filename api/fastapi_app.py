
import io
import base64
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference import PCBSegmentor
from utils import get_device_info


app = FastAPI(
    title="PCB 缺陷分割 API",
    description="PCB 光学质检弱监督分割推理接口",
    version="1.0.0",
)

segmentor: Optional[PCBSegmentor] = None


class DefectDetail(BaseModel):
    """单个缺陷详情"""
    type: str = Field(..., description="缺陷类型: short_circuit/micro_crack/unknown")
    confidence: float = Field(..., description="分类置信度")
    area: int = Field(..., description="缺陷像素面积")
    centroid: dict = Field(default_factory=dict, description="缺陷中心坐标")
    bbox: dict = Field(default_factory=dict, description="缺陷外接矩形")
    aspect_ratio: float = Field(default=0.0, description="长宽比")
    perimeter: float = Field(default=0.0, description="周长")


class ClassificationReport(BaseModel):
    """缺陷分类统计报表"""
    image: dict = Field(default_factory=dict, description="图像信息")
    classification: dict = Field(default_factory=dict, description="图像级分类结果")
    summary: dict = Field(default_factory=dict, description="缺陷汇总统计")
    defect_types: dict = Field(default_factory=dict, description="按类型分类统计")


class InferenceResult(BaseModel):
    """推理结果模型"""
    has_defect: bool = Field(..., description="是否检测到缺陷")
    class_score: float = Field(..., description="图像级分类置信度")
    defect_area: int = Field(..., description="缺陷像素总面积")
    defect_ratio: float = Field(..., description="缺陷面积占比")
    defect_points: list = Field(default_factory=list, description="缺陷坐标点列表")
    image_size: dict = Field(default_factory=dict, description="图像尺寸")
    classification_report: Optional[ClassificationReport] = Field(None, description="缺陷分类统计报表")
    defect_details: Optional[List[DefectDetail]] = Field(None, description="各缺陷详细信息")


class BatchInferenceResponse(BaseModel):
    """批量推理响应"""
    results: List[InferenceResult]
    statistics: dict
    batch_report: Optional[dict] = Field(None, description="批量分类汇总报表")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    device: str
    model_loaded: bool
    device_info: dict


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global segmentor
    checkpoint_path = Path("checkpoints/best.pth")
    config_path = Path("configs/config.yaml")

    if checkpoint_path.exists():
        try:
            segmentor = PCBSegmentor(
                str(checkpoint_path),
                config=str(config_path) if config_path.exists() else None
            )
            print(f"模型已加载: {checkpoint_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            segmentor = None
    else:
        print(f"未找到模型文件: {checkpoint_path}，请调用 /load_model 接口加载")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    device_info = get_device_info()
    return HealthResponse(
        status="ok" if segmentor is not None else "model_not_loaded",
        device=device_info["device"],
        model_loaded=segmentor is not None,
        device_info=device_info,
    )


@app.post("/load_model")
async def load_model(
    checkpoint_path: str = Query(..., description="模型检查点路径"),
    config_path: Optional[str] = Query(None, description="配置文件路径"),
):
    """加载模型"""
    global segmentor

    if not Path(checkpoint_path).exists():
        raise HTTPException(status_code=404, detail=f"模型文件不存在: {checkpoint_path}")

    try:
        segmentor = PCBSegmentor(checkpoint_path, config=config_path)
        return {"status": "success", "message": f"模型加载成功: {checkpoint_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


@app.post("/predict", response_model=InferenceResult)
async def predict(
    file: UploadFile = File(..., description="PCB 图像文件"),
    threshold: Optional[float] = Query(None, description="分割阈值"),
    min_area: Optional[int] = Query(None, description="最小缺陷面积"),
):
    """单图推理接口"""
    if segmentor is None:
        raise HTTPException(status_code=400, detail="模型未加载，请先调用 /load_model")

    try:
        # 读取图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 保存临时文件供推理使用
        temp_path = Path("temp_inference.png")
        image.save(temp_path)

        # 应用参数覆盖
        original_threshold = segmentor.threshold
        original_min_area = segmentor.min_defect_area

        if threshold is not None:
            segmentor.threshold = threshold
        if min_area is not None:
            segmentor.min_defect_area = min_area

        # 推理
        result = segmentor.predict(str(temp_path))

        # 恢复参数
        segmentor.threshold = original_threshold
        segmentor.min_defect_area = original_min_area

        # 清理临时文件
        temp_path.unlink(missing_ok=True)

        # 构造分类统计结果
        classification_report = None
        defect_details = None
        if result.get("classification_report"):
            cr = result["classification_report"]
            classification_report = ClassificationReport(
                image=cr.get("image", {}),
                classification=cr.get("classification", {}),
                summary=cr.get("summary", {}),
                defect_types=cr.get("defect_types", {}),
            )
        if result.get("defect_details"):
            defect_details = [
                DefectDetail(
                    type=d.get("type", "unknown"),
                    confidence=d.get("confidence", 0.0),
                    area=d.get("area", 0),
                    centroid=d.get("centroid", {}),
                    bbox=d.get("bbox", {}),
                    aspect_ratio=d.get("aspect_ratio", 0.0),
                    perimeter=d.get("perimeter", 0.0),
                )
                for d in result["defect_details"]
            ]

        # 返回结果（去掉 numpy 数组）
        return InferenceResult(
            has_defect=result["has_defect"],
            class_score=result["class_score"],
            defect_area=result["defect_area"],
            defect_ratio=result["defect_ratio"],
            defect_points=result["defect_points"],
            image_size=result["image_size"],
            classification_report=classification_report,
            defect_details=defect_details,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.post("/predict/mask")
async def predict_mask(
    file: UploadFile = File(..., description="PCB 图像文件"),
    threshold: Optional[float] = Query(None, description="分割阈值"),
    min_area: Optional[int] = Query(None, description="最小缺陷面积"),
    return_type: str = Query("mask", description="返回类型: mask / overlay / both"),
):
    """获取缺陷掩码图片"""
    if segmentor is None:
        raise HTTPException(status_code=400, detail="模型未加载")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        temp_path = Path("temp_inference_mask.png")
        image.save(temp_path)

        if threshold is not None:
            segmentor.threshold = threshold
        if min_area is not None:
            segmentor.min_defect_area = min_area

        result = segmentor.predict(str(temp_path))
        temp_path.unlink(missing_ok=True)

        if return_type == "overlay":
            from utils.visualize import DefectVisualizer
            visualizer = DefectVisualizer()
            output_img = visualizer.visualize(image, result)
        else:
            output_img = Image.fromarray(result["mask"])

        # 返回图片
        buf = io.BytesIO()
        output_img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(..., description="PCB 图像文件列表"),
    threshold: Optional[float] = Query(None, description="分割阈值"),
    min_area: Optional[int] = Query(None, description="最小缺陷面积"),
):
    """批量推理接口"""
    if segmentor is None:
        raise HTTPException(status_code=400, detail="模型未加载")

    try:
        temp_paths = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            temp_path = Path(f"temp_batch_{len(temp_paths)}.png")
            image.save(temp_path)
            temp_paths.append(temp_path)

        if threshold is not None:
            segmentor.threshold = threshold
        if min_area is not None:
            segmentor.min_defect_area = min_area

        results = segmentor.predict_batch([str(p) for p in temp_paths])

        # 清理临时文件
        for p in temp_paths:
            p.unlink(missing_ok=True)

        # 统计
        total = len(results)
        defect_count = sum(1 for r in results if r["has_defect"])
        statistics = {
            "total_images": total,
            "defect_images": defect_count,
            "normal_images": total - defect_count,
            "defect_ratio": defect_count / total if total > 0 else 0,
        }

        # 生成批量分类汇总报表
        batch_report = None
        if segmentor.enable_classification and segmentor.report_generator is not None:
            batch_report = segmentor.report_generator.generate_batch_report(results)

        # 构造响应
        result_models = []
        for r in results:
            classification_report = None
            defect_details = None
            if r.get("classification_report"):
                cr = r["classification_report"]
                classification_report = ClassificationReport(
                    image=cr.get("image", {}),
                    classification=cr.get("classification", {}),
                    summary=cr.get("summary", {}),
                    defect_types=cr.get("defect_types", {}),
                )
            if r.get("defect_details"):
                defect_details = [
                    DefectDetail(
                        type=d.get("type", "unknown"),
                        confidence=d.get("confidence", 0.0),
                        area=d.get("area", 0),
                        centroid=d.get("centroid", {}),
                        bbox=d.get("bbox", {}),
                        aspect_ratio=d.get("aspect_ratio", 0.0),
                        perimeter=d.get("perimeter", 0.0),
                    )
                    for d in r["defect_details"]
                ]
            result_models.append(InferenceResult(
                has_defect=r["has_defect"],
                class_score=r["class_score"],
                defect_area=r["defect_area"],
                defect_ratio=r["defect_ratio"],
                defect_points=r["defect_points"],
                image_size=r["image_size"],
                classification_report=classification_report,
                defect_details=defect_details,
            ))

        return BatchInferenceResponse(
            results=result_models,
            statistics=statistics,
            batch_report=batch_report,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")


@app.post("/predict/base64")
async def predict_base64(
    image_base64: str = Query(..., description="Base64 编码的图像"),
    threshold: Optional[float] = Query(None),
    min_area: Optional[int] = Query(None),
):
    """Base64 图像推理接口"""
    if segmentor is None:
        raise HTTPException(status_code=400, detail="模型未加载")

    try:
        # 解码 base64
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        temp_path = Path("temp_base64.png")
        image.save(temp_path)

        if threshold is not None:
            segmentor.threshold = threshold
        if min_area is not None:
            segmentor.min_defect_area = min_area

        result = segmentor.predict(str(temp_path))
        temp_path.unlink(missing_ok=True)

        # 生成掩码 base64
        mask_img = Image.fromarray(result["mask"])
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 构造返回（包含分类统计结果）
        response = {
            "has_defect": result["has_defect"],
            "class_score": result["class_score"],
            "defect_area": result["defect_area"],
            "defect_ratio": result["defect_ratio"],
            "defect_points": result["defect_points"],
            "image_size": result["image_size"],
            "mask_base64": mask_base64,
        }
        if result.get("classification_report"):
            response["classification_report"] = result["classification_report"]
        if result.get("defect_details"):
            response["defect_details"] = result["defect_details"]

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
