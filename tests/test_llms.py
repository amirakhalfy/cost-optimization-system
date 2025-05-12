import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from ai_model_crawler import AIModelCrawler
from unittest.mock import patch
from ai_model_crawler import main
from unittest.mock import patch

@pytest.fixture
def crawler():
    """
    Fixture to provide an instance of AIModelCrawler with preconfigured parameters.

    Returns:
        AIModelCrawler: An instance of the crawler with predefined URLs, API token, and output directory.
    """
    return AIModelCrawler(
        urls=["https://example.com/test"],
        api_token="test_token",
        output_dir="./test_output"
    )

@pytest.fixture
def test_data():
    """
    Fixture to provide sample test data for extracting models.

    Returns:
        list: A list of dictionaries representing the extracted model data.
    """
    return [
        {
            "model_name": "TestModel-1",
            "rank": "1",
            "price_or_cost": "$10/1M tokens",
            "arena_score": "95.5",
            "tasks": ["text generation", "summarization"],
            "category": "LLM",
            "organization": "TestOrg"
        },
        {
            "model_name": "TestModel-2",
            "rank": "2",
            "price_or_cost": "$5/1M tokens",
            "provider": "AnotherOrg",
            "ci_95": "±2.3"
        }
    ]

@pytest.mark.asyncio
async def test_crawl_and_extract_success(crawler, test_data):
    """
    Test the crawl_and_extract method for successful data extraction and saving.

    This test mocks the AsyncWebCrawler, processes a URL, and verifies that the
    output file is saved correctly. It also ensures that the method calls
    to _process_url and save_to_json occur as expected.
    """
    with patch('ai_model_crawler.AsyncWebCrawler') as mock_crawler:
        mock_crawler_instance = AsyncMock()
        mock_crawler.return_value.__aenter__.return_value = mock_crawler_instance

        crawler._process_url = AsyncMock(return_value=test_data)

        crawler.save_to_json = MagicMock(return_value="/path/to/output.json")

        with patch('ai_model_crawler.litellm', create=True):
            output_file = await crawler.crawl_and_extract()

            assert output_file == "/path/to/output.json"
            crawler._process_url.assert_called_once()
            crawler.save_to_json.assert_called_once_with(test_data)

@pytest.mark.asyncio
async def test_retry_with_timeout_success(crawler):
    """
    Test the retry_with_timeout method for a successful function execution.

    This test mocks a function that always succeeds and ensures that it is
    retried only once before returning the expected result.
    """
    mock_func = AsyncMock(return_value="success")
    result = await crawler._retry_with_timeout(mock_func, retries=3, delay=0.1)
    assert result == "success"
    assert mock_func.call_count == 1

@pytest.mark.asyncio
async def test_retry_with_timeout_failure(crawler):
    """
    Test the retry_with_timeout method for failure in function execution.

    This test mocks a function that always fails and ensures that the retries
    occur until the maximum limit is reached, and the function returns None after
    all retries are exhausted.
    """
    mock_func = AsyncMock(side_effect=Exception("Always fail"))
    result = await crawler._retry_with_timeout(mock_func, retries=2, delay=0.1)
    assert result is None
    assert mock_func.call_count == 2

def test_process_extracted_data(crawler, test_data):
    """
    Test the _process_extracted_data method for correct data processing.

    This test ensures that the extracted data is processed into the correct format
    and that the expected keys and values are present in the processed data.
    """
    processed = crawler._process_extracted_data(test_data)
    
    assert len(processed) == 2
    
    first_model = processed[0]
    assert first_model.get("benchmarks", {}).get("arena_score") == "95.5"
    assert first_model.get("provider") == "TestOrg"
    assert first_model.get("tasks") == ["text generation", "summarization"]
    
    second_model = processed[1]
    assert second_model.get("benchmarks", {}).get("ci_95") == "±2.3"
    assert second_model.get("provider") == "AnotherOrg"

def test_save_to_json(crawler, test_data, tmp_path):
    """
    Test the save_to_json method for correctly saving extracted data to a JSON file.

    This test ensures that the extracted data is saved to a JSON file in the output directory
    and that the file content matches the expected structure.
    """
    crawler.output_dir = tmp_path
    filename = crawler.save_to_json(test_data, "test_output.json")
    
    file_path = tmp_path / "test_output.json"
    assert file_path.exists()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    assert len(saved_data) == 2
    assert saved_data[0]["model_name"] == "TestModel-1"

def test_create_llm_strategy(crawler):
    """
    Test the _create_llm_strategy method for generating a valid strategy.

    This test ensures that the LLM strategy is created with a valid schema and
    that the strategy contains the expected properties.
    """
    strategy = crawler._create_llm_strategy()
    assert strategy is not None
    
    assert isinstance(strategy.schema, dict)
    assert "properties" in strategy.schema
    assert "model_name" in strategy.schema["properties"]

@pytest.mark.asyncio
async def test_main_success():
    """
    Test the main function for successful execution.

    This test mocks the AIModelCrawler and verifies that the crawl_and_extract method
    is called correctly and the main function returns the expected result.
    """
    with patch('ai_model_crawler.AIModelCrawler') as mock_crawler_class:
        mock_crawler = AsyncMock()
        mock_crawler.crawl_and_extract.return_value = "/path/to/output.json"
        mock_crawler_class.return_value = mock_crawler
                
        result = await main()
        
        assert result == 0
        mock_crawler.crawl_and_extract.assert_called_once()
