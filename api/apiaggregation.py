from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional
from pydantic import BaseModel
from typing import Dict
from app.db.db_setup import get_db
from app.db.models import Model, Provider, Pricing, Benchmark, Task, ModelTask, BenchmarkResult

router = APIRouter(prefix="/aggregation")

class ModelComparison(BaseModel):
    model_id: int
    model_name: str
    provider_name: str
    input_cost: float
    output_cost: float
    benchmark_scores: dict
    performance_to_cost_ratio: Optional[float] = None
    tasks: List[str]
    context_window: Optional[int] = None
    parameters: Optional[int] = None

class CostPerformanceMetric(BaseModel):
    model_id: int
    model_name: str
    provider_name: str
    benchmark_name: str
    score: float
    cost_per_point: float
    cost_efficiency_rank: int

class ModelRecommendation(BaseModel):
    task_name: str
    best_overall_model: dict
    best_budget_model: dict
    best_performance_model: dict

@router.get("/compare-models", response_model=List[ModelComparison])
def compare_models(
    model_ids: List[int] = Query(..., description="List of model IDs to compare"),
    benchmark_id: Optional[int] = None,
    task_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Compare multiple AI models based on pricing, benchmark performance, and supported tasks.

    This endpoint retrieves detailed comparison data for multiple models, including:
    - Provider information
    - Latest input/output pricing
    - Benchmark scores (optionally filtered by benchmark ID)
    - Supported tasks (optionally filtered by task ID)
    - A performance-to-cost ratio based on benchmark averages and pricing

    Parameters:
    ----------
    model_ids : List[int]
        List of model IDs to include in the comparison. Required.
    benchmark_id : Optional[int]
        (Optional) If provided, benchmark scores are filtered to this benchmark.
    task_id : Optional[int]
        (Optional) If provided, only tasks matching this task ID will be included per model.
    db : Session
        SQLAlchemy database session injected by FastAPI dependency.

    Returns:
    -------
    List[ModelComparison]
        A list of model comparison objects including pricing, performance, and metadata.
    
    Raises:
    ------
    HTTPException (404):
        If a model with the given ID does not exist.
    """
    models_data = []

    for model_id in model_ids:
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")

        provider = db.query(Provider).filter(Provider.id == model.provider_id).first()

        pricing = db.query(Pricing).filter(Pricing.model_id == model_id).order_by(Pricing.created_at.desc()).first()

        benchmark_query = db.query(BenchmarkResult, Benchmark.name).join(
            Benchmark, BenchmarkResult.benchmark_id == Benchmark.id
        ).filter(BenchmarkResult.model_id == model_id)

        if benchmark_id:
            benchmark_query = benchmark_query.filter(BenchmarkResult.benchmark_id == benchmark_id)

        benchmark_results = benchmark_query.all()
        benchmark_scores = {b_name: result.score for (result, b_name) in benchmark_results}

        task_query = db.query(Task.task_name).join(
            ModelTask, Task.id == ModelTask.task_id
        ).filter(ModelTask.model_id == model_id)

        if task_id:
            task_query = task_query.filter(Task.id == task_id)

        tasks = [task[0] for task in task_query.all()]

        performance_to_cost_ratio = None
        if benchmark_scores and pricing:
            avg_score = sum(benchmark_scores.values()) / len(benchmark_scores)
            avg_cost = (pricing.input_cost + pricing.output_cost) / 2
            if avg_cost > 0:
                performance_to_cost_ratio = avg_score / avg_cost

        models_data.append(ModelComparison(
            model_id=model.id,
            model_name=model.name,
            provider_name=provider.name if provider else "Unknown",
            input_cost=pricing.input_cost if pricing else 0,
            output_cost=pricing.output_cost if pricing else 0,
            benchmark_scores=benchmark_scores,
            performance_to_cost_ratio=performance_to_cost_ratio,
            tasks=tasks,
            context_window=model.context_window,
            parameters=model.parameters
        ))

    return models_data


@router.get("/provider-comparison")
def compare_providers(db: Session = Depends(get_db)):
    """
    Aggregate and compare AI model providers based on their model offerings, pricing, supported tasks, and benchmark performance.

    This endpoint provides a high-level summary for each provider, including:
    - Number of models offered
    - Average input and output costs across all models
    - Average benchmark performance per benchmark category
    - List of distinct tasks supported by the provider's models

    This information is useful for evaluating providers on performance-to-cost efficiency, model diversity, and suitability for specific tasks.

    Returns:
    --------
    List[dict]
        A list of dictionaries where each entry summarizes a provider's capabilities and offerings.

    Example Output:
    ---------------
    [
        {
            "provider_id": 1,
            "provider_name": "OpenAI",
            "model_count": 4,
            "avg_input_cost": 0.002,
            "avg_output_cost": 0.004,
            "benchmark_performance": {
                "MMLU": 78.5,
                "HumanEval": 65.1
            },
            "tasks_covered": ["translation", "code-generation", "chat"]
        },
        ...
    ]
    """
    providers = db.query(Provider).all()
    
    result = []
    for provider in providers:
        models = db.query(Model).filter(Model.provider_id == provider.id).all()
        model_ids = [model.id for model in models]
        
        if not model_ids:
            continue
            
        avg_pricing = db.query(
            func.avg(Pricing.input_cost).label("avg_input_cost"),
            func.avg(Pricing.output_cost).label("avg_output_cost")
        ).filter(
            Pricing.model_id.in_(model_ids)
        ).first()
        
        benchmark_performance = db.query(
            Benchmark.name,
            func.avg(BenchmarkResult.score).label("avg_score")
        ).join(
            BenchmarkResult, Benchmark.id == BenchmarkResult.benchmark_id
        ).filter(
            BenchmarkResult.model_id.in_(model_ids)
        ).group_by(
            Benchmark.name
        ).all()
        
        tasks_covered = db.query(
            Task.task_name
        ).join(
            ModelTask, Task.id == ModelTask.task_id
        ).filter(
            ModelTask.model_id.in_(model_ids)
        ).distinct().all()
        
        result.append({
            "provider_id": provider.id,
            "provider_name": provider.name,
            "model_count": len(models),
            "avg_input_cost": avg_pricing.avg_input_cost if avg_pricing.avg_input_cost else 0,
            "avg_output_cost": avg_pricing.avg_output_cost if avg_pricing.avg_output_cost else 0,
            "benchmark_performance": {name: score for name, score in benchmark_performance},
            "tasks_covered": [task[0] for task in tasks_covered]
        })
    
    return result

@router.get("/provider-info/{provider_name}", response_model=Dict)
def get_provider_info_by_name(provider_name: str, db: Session = Depends(get_db)):
    """
    Retrieve detailed information about a provider's models, pricing, benchmarks, and supported tasks.

    Parameters:
    ----------
    provider_name : str
        The name of the provider (e.g., "OpenAI", "DeepSeek-AI", "Meta").
    db : Session
        The database session.

    Returns:
    -------
    dict
        A dictionary containing provider's model details, average pricing, benchmark performance, and supported tasks.

    Raises:
    ------
    HTTPException (404)
        If a provider with the given name does not exist.
    """
    provider = db.query(Provider).filter(Provider.name == provider_name).first()
    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider with name {provider_name} not found")
    
    models = db.query(Model).filter(Model.provider_id == provider.id).all()
    model_ids = [model.id for model in models]
    
    if not model_ids:
        raise HTTPException(status_code=404, detail=f"No models found for provider {provider_name}")
    
    avg_pricing = db.query(
        func.avg(Pricing.input_cost).label("avg_input_cost"),
        func.avg(Pricing.output_cost).label("avg_output_cost")
    ).filter(Pricing.model_id.in_(model_ids)).first()

    benchmark_performance = db.query(
        Benchmark.name,
        func.avg(BenchmarkResult.score).label("avg_score")
    ).join(
        BenchmarkResult, Benchmark.id == BenchmarkResult.benchmark_id
    ).filter(
        BenchmarkResult.model_id.in_(model_ids)
    ).group_by(
        Benchmark.name
    ).all()

    tasks_covered = db.query(
        Task.task_name
    ).join(
        ModelTask, Task.id == ModelTask.task_id
    ).filter(
        ModelTask.model_id.in_(model_ids)
    ).distinct().all()

    provider_info = {
        "provider_name": provider.name,
        "model_count": len(models),
        "avg_input_cost": avg_pricing.avg_input_cost if avg_pricing.avg_input_cost else 0,
        "avg_output_cost": avg_pricing.avg_output_cost if avg_pricing.avg_output_cost else 0,
        "benchmark_performance": {name: score for name, score in benchmark_performance},
        "tasks_covered": [task[0] for task in tasks_covered]
    }

    return provider_info



@router.get("/best-value-models", response_model=List[CostPerformanceMetric])
def get_best_value_models(
    benchmark_name: str,  
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    This function retrieves and ranks AI models based on their cost-efficiency,
    calculated as the average cost per performance score, for a specific benchmark.
    It returns the top models, allowing users to identify the most cost-effective options for a given 
    benchmark.

    **Parameters**:
    - `benchmark_name`: The name of the benchmark to filter the models by.
    - `limit`: (Optional) The number of models to return. Defaults to 10.

    **Returns**:
    - A list of models ranked by their cost-efficiency (performance per cost ratio) for the specified benchmark.
    Each model is returned with the following information:
    - `model_id`: The unique ID of the model.
    - `model_name`: The name of the model.
    - `provider_name`: The name of the provider offering the model.
    - `benchmark_name`: The name of the benchmark used for evaluation.
    - `score`: The score the model achieved in the benchmark.
    - `cost_per_point`: The cost per point of performance, calculated by dividing the average cost by the benchmark score.
    - `cost_efficiency_rank`: The rank of the model based on its cost-efficiency, where rank 1 is the best.

    **Raises**:
    - HTTPException(404): If no models are found for the specified `benchmark_name`.
    - HTTPException(500): If there is an error in the database query.
    """
    results = db.query(
        Model.id.label("model_id"),
        Model.name.label("model_name"),
        Provider.name.label("provider_name"),
        Benchmark.name.label("benchmark_name"),
        BenchmarkResult.score.label("score"),
        Pricing.input_cost,
        Pricing.output_cost
    ).join(
        Provider, Model.provider_id == Provider.id
    ).join(
        BenchmarkResult, Model.id == BenchmarkResult.model_id
    ).join(
        Benchmark, BenchmarkResult.benchmark_id == Benchmark.id
    ).join(
        Pricing, Model.id == Pricing.model_id
    ).filter(
        Benchmark.name == benchmark_name  
    ).all()

    if not results:
        raise HTTPException(status_code=404, detail="No models found for the given benchmark")

    metrics = []
    for r in results:
        if r.input_cost > 0 and r.output_cost > 0:
            avg_cost = (r.input_cost + r.output_cost) / 2
            cost_per_point = avg_cost / r.score if r.score > 0 else float('inf')
        else:
            cost_per_point = float('inf')

        metrics.append({
            "model_id": r.model_id,
            "model_name": r.model_name,
            "provider_name": r.provider_name,
            "benchmark_name": r.benchmark_name,
            "score": r.score,
            "cost_per_point": cost_per_point,
            "cost_efficiency_rank": 0 
        })
    
    metrics.sort(key=lambda x: x["cost_per_point"])
    
    for i, metric in enumerate(metrics):
        metric["cost_efficiency_rank"] = i + 1

    return metrics[:limit]


@router.get("/model-recommendations", response_model=List[ModelRecommendation])
def get_model_recommendations(
    task_ids: List[int] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Retrieve model recommendations for specified tasks. For each task, it returns three recommendations:
    1. The best overall model (based on the value score).
    2. The best budget model (lowest average cost with a non-zero average score).
    3. The best performance model (highest benchmark score).

    Args:
        task_ids (List[int], optional): A list of task IDs to filter the recommendations by. If not provided, 
                                         recommendations for all tasks will be returned. Defaults to None.
        db (Session, required): A database session dependency that allows interaction with the database.

    Returns:
        List[ModelRecommendation]: A list of model recommendations, with each entry containing the task name, 
                                   the best overall model, the best budget model, and the best performance model.
    
    Example:
        {
            "task_name": "Task A",
            "best_overall_model": {
                "model_id": 1,
                "model_name": "Model X",
                "provider": "Provider A",
                "avg_cost": 10.0,
                "avg_score": 85.0,
                "value_score": 8.5
            },
            "best_budget_model": {
                "model_id": 2,
                "model_name": "Model Y",
                "provider": "Provider B",
                "avg_cost": 5.0,
                "avg_score": 80.0,
                "value_score": 16.0
            },
            "best_performance_model": {
                "model_id": 3,
                "model_name": "Model Z",
                "provider": "Provider C",
                "avg_cost": 15.0,
                "avg_score": 90.0,
                "value_score": 6.0
            }
        }
    """
    recommendations = []
    
    tasks_query = db.query(Task)
    if task_ids:
        tasks_query = tasks_query.filter(Task.id.in_(task_ids))
    
    tasks = tasks_query.all()
    
    for task in tasks:
        models = db.query(
            Model.id.label("model_id"),
            Model.name.label("model_name"),
            Provider.name.label("provider_name"),
            Pricing.input_cost,
            Pricing.output_cost,
            func.avg(BenchmarkResult.score).label("avg_score")
        ).join(
            Provider, Model.provider_id == Provider.id
        ).join(
            ModelTask, Model.id == ModelTask.model_id
        ).join(
            Pricing, Model.id == Pricing.model_id
        ).outerjoin(
            BenchmarkResult, Model.id == BenchmarkResult.model_id
        ).filter(
            ModelTask.task_id == task.id
        ).group_by(
            Model.id
        ).all()
        
        best_overall = None
        best_budget = None
        best_performance = None
        
        model_metrics = []
        for m in models:
            avg_cost = (m.input_cost + m.output_cost) / 2
            value_score = m.avg_score / avg_cost if m.avg_score and avg_cost > 0 else 0
            
            model_metrics.append({
                "model_id": m.model_id,
                "model_name": m.model_name,
                "provider": m.provider_name,
                "avg_cost": avg_cost,
                "avg_score": m.avg_score or 0,
                "value_score": value_score
            })
        
        if model_metrics:
            best_overall = max(model_metrics, key=lambda x: x["value_score"])
            
            qualifying_models = [m for m in model_metrics if m["avg_score"] > 0]
            if qualifying_models:
                best_budget = min(qualifying_models, key=lambda x: x["avg_cost"])
            
            best_performance = max(model_metrics, key=lambda x: x["avg_score"])
        
        recommendations.append(ModelRecommendation(
            task_name=task.task_name,
            best_overall_model=best_overall or {},
            best_budget_model=best_budget or {},
            best_performance_model=best_performance or {}
        ))
    
    return recommendations

@router.get("/cost-optimizer")
def optimize_cost(
    task_id: int,
    min_benchmark_score: Optional[float] = None,
    max_budget: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """
    Find the most cost-effective model for a specific task that meets minimum benchmark requirements
    and stays within budget constraints.
    """
    query = db.query(
        Model.id,
        Model.name,
        Provider.name.label("provider"),
        Pricing.input_cost,
        Pricing.output_cost,
        func.avg(BenchmarkResult.score).label("avg_benchmark_score")
    ).join(
        Provider, Model.provider_id == Provider.id
    ).join(
        ModelTask, Model.id == ModelTask.model_id
    ).join(
        Pricing, Model.id == Pricing.model_id
    ).outerjoin(
        BenchmarkResult, Model.id == BenchmarkResult.model_id
    ).filter(
        ModelTask.task_id == task_id
    ).group_by(
        Model.id
    )
    
    if min_benchmark_score is not None:
        query = query.having(func.avg(BenchmarkResult.score) >= min_benchmark_score)
    
    models = query.all()
    
    results = []
    for model in models:
        total_cost = model.input_cost + model.output_cost
        
        if max_budget is not None and total_cost > max_budget:
            continue
            
        efficiency = model.avg_benchmark_score / total_cost if total_cost > 0 else 0
        
        results.append({
            "model_id": model.id,
            "model_name": model.name,
            "provider": model.provider,
            "input_cost": model.input_cost,
            "output_cost": model.output_cost,
            "total_cost": total_cost,
            "avg_benchmark_score": model.avg_benchmark_score,
            "cost_efficiency": efficiency
        })
    
    results.sort(key=lambda x: x["cost_efficiency"], reverse=True)
    
    return {
        "task_id": task_id,
        "optimized_models": results
    }

@router.get("/benchmark-leaderboard")
def get_benchmark_leaderboard(
    benchmark_id: int,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get a leaderboard of models ranked by their performance on a specific benchmark.
    """
    benchmark = db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    results = db.query(
        Model.id,
        Model.name,
        Provider.name.label("provider"),
        BenchmarkResult.score
    ).join(
        Provider, Model.provider_id == Provider.id
    ).join(
        BenchmarkResult, Model.id == BenchmarkResult.model_id
    ).filter(
        BenchmarkResult.benchmark_id == benchmark_id
    ).order_by(
        desc(BenchmarkResult.score)
    ).limit(limit).all()
    
    return {
        "benchmark_id": benchmark_id,
        "benchmark_name": benchmark.name,
        "leaderboard": [
            {
                "rank": i + 1,
                "model_id": result.id,
                "model_name": result.name,
                "provider": result.provider,
                "score": result.score
            } for i, result in enumerate(results)
        ]
    }

@router.get("/task-capability-matrix")
def get_task_capability_matrix(db: Session = Depends(get_db)):
    """
    Generate a matrix showing which models support which tasks.
    """
    tasks = db.query(Task).all()
    task_ids = [task.id for task in tasks]
    
    models = db.query(
        Model.id,
        Model.name,
        Provider.name.label("provider")
    ).join(
        Provider, Model.provider_id == Provider.id
    ).all()
    
    matrix = []
    for model in models:
        supported_tasks = db.query(
            Task.id
        ).join(
            ModelTask, Task.id == ModelTask.task_id
        ).filter(
            ModelTask.model_id == model.id
        ).all()
        
        supported_task_ids = [task[0] for task in supported_tasks]
        
        capabilities = {
            task.id: task.id in supported_task_ids 
            for task in tasks
        }
        
        matrix.append({
            "model_id": model.id,
            "model_name": model.name,
            "provider": model.provider,
            "task_capabilities": capabilities,
            "supported_task_count": len(supported_task_ids)
        })
    
    matrix.sort(key=lambda x: x["supported_task_count"], reverse=True)
    
    return {
        "tasks": [{"id": task.id, "name": task.task_name} for task in tasks],
        "models": matrix
    }
@router.get("/open-source-models", response_model=List[dict])
def get_open_source_models(
    task_id: Optional[int] = None,
    benchmark_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve models that are likely open-source (free to use) based on pricing data.
    
    This endpoint identifies models that either:
    1. Have no pricing data (NULL input_cost and output_cost)
    2. Have zero pricing (input_cost=0 and output_cost=0)
    
    These criteria typically indicate open-source models that can be run locally
    without API costs.
    
    Parameters:
    ----------
    task_id : Optional[int]
        If provided, filters models to those supporting this specific task.
    benchmark_id : Optional[int]
        If provided, includes benchmark scores for this specific benchmark.
    db : Session
        SQLAlchemy database session.
        
    Returns:
    -------
    List[dict]
        A list of open-source models with their details including:
        - model_id: The unique identifier for the model
        - model_name: The name of the model
        - provider_name: The organization that created the model
        - parameters: Number of parameters (model size)
        - context_window: Maximum context window size
        - tasks: List of tasks the model supports
        - benchmark_scores: Performance scores if benchmark_id is specified    
    
   
    """
    query = db.query(
        Model.id.label("model_id"),
        Model.name.label("model_name"),
        Provider.name.label("provider_name"),
        Model.parameters,
        Model.context_window
    ).join(
        Provider, Model.provider_id == Provider.id
    ).outerjoin(
        Pricing, Model.id == Pricing.model_id
    ).filter(
        (Pricing.id == None) |
        ((Pricing.input_cost == 0) & (Pricing.output_cost == 0))
    )
    
    if task_id is not None:
        query = query.join(
            ModelTask, Model.id == ModelTask.model_id
        ).filter(
            ModelTask.task_id == task_id
        )
    
    models = query.all()
    result = []
    
    for model in models:
        task_query = db.query(Task.task_name).join(
            ModelTask, Task.id == ModelTask.task_id
        ).filter(
            ModelTask.model_id == model.model_id
        )
        tasks = [task[0] for task in task_query.all()]
        
        benchmark_scores = {}
        if benchmark_id is not None:
            benchmark_results = db.query(
                Benchmark.name,
                BenchmarkResult.score
            ).join(
                BenchmarkResult, Benchmark.id == BenchmarkResult.benchmark_id
            ).filter(
                BenchmarkResult.model_id == model.model_id,
                BenchmarkResult.benchmark_id == benchmark_id
            ).all()
            
            benchmark_scores = {name: score for name, score in benchmark_results}
        
        result.append({
            "model_id": model.model_id,
            "model_name": model.model_name,
            "provider_name": model.provider_name,
            "parameters": model.parameters,
            "context_window": model.context_window,
            "tasks": tasks,
            "benchmark_scores": benchmark_scores
        })
    
    return result