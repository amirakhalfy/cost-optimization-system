from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app  
client = TestClient(app)


def test_compare_models():
    """This test makes sure the retrieval of models is succesful. (API return status code == 200)"""
    response = client.get("/aggregation/compare-models", params={"model_ids": [94,95]})
    
    assert response.status_code == 200
    
    assert isinstance(response.json(), list)
    
    assert "model_id" in response.json()[0]


def test_provider_comparison():
    """This test makes sure the retrieval of AI Provider comparison. (API returns status code == 200)"""
    response = client.get("/aggregation/provider-comparison")
    
    assert response.status_code == 200
    
    assert isinstance(response.json(), list)


def test_get_best_value_models():
    """This test covers the evaluation of models. The API must have a return status of 200 (Works)
    The API response must contain : the evaluated model ID (model_id), the cost per point (cost_per_point) of the model and the rank of the model (cost_efficiency_rank)"""
    response = client.get("/aggregation/best-value-models", params={"benchmark_name": "MMLU"})
    
    assert response.status_code == 200
    
    assert isinstance(response.json(), list)
    
    if response.json():
        assert "model_id" in response.json()[0]
        assert "cost_per_point" in response.json()[0]
        assert "cost_efficiency_rank" in response.json()[0]


def test_provider_info_by_name():
    """This test uses the example of the OpenAI LLM provider to make sure the information 
    retrieved about the said provider contains the provider name (provider_name), 
    the number of models (model_count), the model performance (benchmark_performance), 
    and the model tasks. (tasks_covered)
    
    It covers : valid provider, non-existent provider (Response Code == 404).

    Expected Output (Existent Provider) :
    -provider_name : the name of the LLM provider.
    -model_count : the number of models.
    -benchmark_performance : the performance of the model.
    -tasks_covered : the tasks covered by the model.
    -Status Code == 200

    Expected Output (NonExistent Provider) :
    -Status Code == 404
    """
    response = client.get("/aggregation/provider-info/OpenAI")
    
    assert response.status_code == 200
    
    result = response.json()
    assert "provider_name" in result
    assert "model_count" in result
    assert "benchmark_performance" in result
    assert "tasks_covered" in result
    
    response = client.get("/aggregation/provider-info/NonExistentProvider")
    assert response.status_code == 404


def test_model_recommendations():
    """
    This test covers the behaviour of the recommendation provided for the inputted task.
    Expected Output : 
    -task_name : The task required of the LLM.
    -best_overall_model : The best model for the task.
    -best_budget_model : The model with the best cost.
    -best_performance_model : The model with the best performance.
    -Response Code : 200
    """
    response = client.get("/aggregation/model-recommendations", params={"task_ids": [1, 2]})
    
    assert response.status_code == 200
    
    assert isinstance(response.json(), list)
    
    if response.json():
        recommendation = response.json()[0]
        assert "task_name" in recommendation
        assert "best_overall_model" in recommendation
        assert "best_budget_model" in recommendation
        assert "best_performance_model" in recommendation

#################### TESTS TO IMPLEMENT AFTER FIXING BUDGET #######################

# def test_cost_optimizer():
#     """
#     """
#     response = client.get("/aggregation/cost-optimizer", params={"task_id": 1})
    
#     assert response.status_code == 200
    
#     result = response.json()
#     assert "task_id" in result
#     assert "optimized_models" in result
#     assert isinstance(result["optimized_models"], list)
    
#     response = client.get("/aggregation/cost-optimizer", 
#                          params={"task_id": 1, 
#                                 "min_benchmark_score": 70.0, 
#                                 "max_budget": 0.01})
    
#     assert response.status_code == 200
    
#     result = response.json()
#     for model in result["optimized_models"]:
#         assert model["avg_benchmark_score"] >= 70.0
#         assert model["total_cost"] <= 0.01


# def test_benchmark_leaderboard():
#     response = client.get("/aggregation/benchmark-leaderboard", params={"benchmark_id": 1})
    
#     assert response.status_code == 200
    
#     result = response.json()
#     assert "benchmark_id" in result
#     assert "benchmark_name" in result
#     assert "leaderboard" in result
    
#     if result["leaderboard"]:
#         entry = result["leaderboard"][0]
#         assert "rank" in entry
#         assert "model_id" in entry
#         assert "model_name" in entry
#         assert "score" in entry
    
    # response = client.get("/aggregation/benchmark-leaderboard", params={"benchmark_id": 9999})
    # assert response.status_code == 404


# def test_task_capability_matrix():
#     response = client.get("/aggregation/task-capability-matrix")
    
#     assert response.status_code == 200
    
#     result = response.json()
#     assert "tasks" in result
#     assert "models" in result
    
#     if result["models"]:
#         model = result["models"][0]
#         assert "model_id" in model
#         assert "model_name" in model
#         assert "task_capabilities" in model
#         assert "supported_task_count" in model