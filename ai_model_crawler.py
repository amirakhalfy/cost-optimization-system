import asyncio
import json
import os
from dotenv import load_dotenv
import re
import random
from typing import List, Optional, Dict, Any, Callable, Set
from datetime import datetime
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, ContentRelevanceFilter, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from pydantic import BaseModel, Field
from app.services.savingdb import save_to_database
load_dotenv()

class LocalLLMConfig:
    def __init__(self, provider: str, api_token: str = "", base_url: str = ""):
        self.provider = provider
        self.api_token = api_token
        self.base_url = base_url

class AIModel(BaseModel):
    """
    AI Model schema used for data extraction and storage in line with SQL tables.
    """

    model_name: str = Field(description="The name of the AI model")
    provider: Optional[str] = Field(description="The name of the model provider", default=None)
    provider_id: Optional[int] = Field(description="The ID of the provider (foreign key)", default=None)
    license: Optional[str] = Field(description="License details of the model", default=None)
    description: Optional[str] = Field(description="A brief description of the model", default=None)
    context_window: Optional[int] = Field(description="The context window size for the model", default=None)
    max_tokens: Optional[int] = Field(description="The maximum token limit for the model", default=None)
    benchmarks: Optional[Dict[str, float]] = Field(description="Performance metrics (e.g., Average, MMLU, etc.)", default=None)
    tasks: Optional[List[str]] = Field(description="Tasks the model is designed for", default=None)
    parameters: Optional[int] = Field(description="Number of parameters in the model", default=None)
    pricing: Optional[Dict[str, Any]] = Field(description="Pricing information for the model", default=None)
    

class AIModelCrawler:
    """
    A crawler to scrape AI model data from given URLs and extract relevant details.

    Attributes:
        urls (List[str]): The URLs of websites to scrape.
        api_token (str): The API token for accessing the AI model data.
        output_dir (str): The directory to save the extracted data.
    """

    EXTRACTION_INSTRUCTION = """
    Extract AI model information from the content, focusing on models mentioned on the page. 
    Look for information in tables, text descriptions, and pricing sections.
    For each AI model found, extract:
    - model_name: Name of the AI model (e.g., GPT-4, Claude 3 Opus)
    - provider: Organization providing the model (e.g., OpenAI, Anthropic)
    - license: License details if available
    - description: Brief description of the model's capabilities
    - context_window: Context window size in tokens (integer)
    - max_tokens: Maximum tokens the model can process (integer)
    - parameters: Number of parameters - convert values like "70B" to integer 70
    - benchmarks: Performance metrics like MMLU, GPQA, HumanEval, etc.   
      - name: Name of the benchmark for example name humanEval : 
      - score_benchmark: Numerical score achieved for the specific benchmark
    - confidence_interval: Average 95% confidence interval (if available) is a seperated field in the benchmark and it should be a string 
    - votes: Average number of votes (if available) in the benchmark
    - tasks: List of tasks the model is designed for
    - pricing: Extract pricing information including input_cost, output_cost,cached_input, training_cost, unit , and currency
    -rank : the model rank should be saved in the benchmark
    Extract ALL models mentioned on the page, even if information is incomplete.
    """

    MODEL_KEYWORDS = [
        "llm", "language model", "ai model", "gpt", "claude", "llama", "mistral",
        "gemini", "leaderboard", "benchmark", "context window", "parameters", 
        "model pricing", "token pricing", "api access"
    ]

    def __init__(self, urls: List[str], api_token: str = "", output_dir: str = "./output", 
                 delay_between_urls: int = 60, retry_base_delay: int = 15):
        """
        Initialize the AIModelCrawler instance.

        Args:
            urls (List[str]): The URLs of websites to scrape.
            api_token (str, optional): The API token for accessing the data. Defaults to an empty string.
            output_dir (str, optional): The directory to store the output. Defaults to './output'.
            delay_between_urls (int, optional): Delay in seconds between processing URLs. Defaults to 60.
            retry_base_delay (int, optional): Base delay for retry mechanism. Defaults to 15.
        """
        self.urls = urls if isinstance(urls, list) else [urls]
        self.api_token = api_token
        self.output_dir = output_dir
        self.seen_models: Set[str] = set()  
        self.delay_between_urls = delay_between_urls
        self.retry_base_delay = retry_base_delay
        
        os.makedirs(self.output_dir, exist_ok=True)

    def _create_llm_strategy(self) -> LLMExtractionStrategy:
        """
        Create and return an LLM extraction strategy.

        Returns:
            LLMExtractionStrategy: The extraction strategy configured for AI model data.
        """
        return LLMExtractionStrategy(
            llm_config=LocalLLMConfig(
            provider="groq/meta-llama/llama-4-scout-17b-16e-instruct",
            api_token=self.api_token,
            base_url=""
            ),
            schema=AIModel.model_json_schema(),
            extraction_type="schema",
            instruction=self.EXTRACTION_INSTRUCTION,
            chunk_token_threshold=1000,  
            overlap_rate=0.1,
            apply_chunking=True,
            input_format="markdown",
            extra_args={ 
                "temperature":0.05,
                "max_tokens":5000,  
            },
        )

    def _create_deep_crawl_strategy(self):
        """
        Create a deep crawl strategy for finding relevant LLM information.

        Returns:
            BestFirstCrawlingStrategy: The configured deep crawl strategy.
        """
        filter_chain = FilterChain([
            URLPatternFilter(patterns=[
                "*model*", "*benchmark*", "*leaderboard*", "*pricing*", "*api*",
                "*llm*", "*service*", "*product*", "*specification*"
            ]),
            
            ContentRelevanceFilter(
                query="AI language model specifications pricing parameters benchmarks",
                threshold=0.3
            )
        ])
        
        keyword_scorer = KeywordRelevanceScorer(
            keywords=self.MODEL_KEYWORDS,
            weight=0.8
        )
        
        return BestFirstCrawlingStrategy(
            max_depth=2,             
            include_external=False,  
            max_pages=2,             
            filter_chain=filter_chain,
            url_scorer=keyword_scorer
        )

    async def _retry_with_timeout(self, func: Callable, retries: int = 5, 
                                 delay: int = None, max_delay: int = 120):
        """
        Retry mechanism with improved error handling and exponential backoff.

        Args:
            func (Callable): The function to execute with retries.
            retries (int, optional): The number of retries before giving up. Defaults to 5.
            delay (int, optional): The initial delay between retries in seconds. Defaults to retry_base_delay.
            max_delay (int, optional): Maximum delay between retries. Defaults to 120 seconds.

        Returns:
            Any: The result of the function if successful, or None after retries.
        """
        delay = delay or self.retry_base_delay
        
        for attempt in range(retries):
            try:
                return await func()
            except asyncio.TimeoutError:
                print(f"Attempt {attempt + 1} failed with timeout.")
            except Exception as e:
                print(f"Error in attempt {attempt + 1}: {str(e)}")
            
            backoff_time = min(delay * (2 ** attempt) + random.uniform(0, 5), max_delay)
            print(f"Retrying in {backoff_time:.1f} seconds (attempt {attempt+1}/{retries})...")
            await asyncio.sleep(backoff_time)
            
        print("All retries failed.")
        return None

    def save_to_json(self, data, filename=None):
        """
        Save the extracted data to a JSON file.

        Args:
            data (Any): The data to be saved.
            filename (str, optional): The filename for the output JSON. Defaults to None.

        Returns:
            str: The path to the saved JSON file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_models_extracted_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to {filepath}")
        return filepath
    
    def _normalize_model_name(self, name):
        """
        Normalize model names for better comparison and deduplication.
        
        Args:
            name (str): The model name to normalize
            
        Returns:
            str: The normalized model name
        """
        if not name:
            return ""
        
        normalized = name.lower().strip()
        
        normalized = re.sub(r'[^\w\s\-]', '', normalized)
        
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized

    def _extract_pricing_info(self, item):
        """
        Extract and standardize pricing information from model data.
        
        Args:
            item (dict): The model data item
            
        Returns:
            dict: Updated model data with standardized pricing
        """
        pricing = {}
        
        for key in ['input_cost', 'output_cost', 'price_per_token', 'price']:
            if key in item and item[key]:
                if isinstance(item[key], str):
                    price_str = item[key]
                    currency_match = re.search(r'[$€£¥]', price_str)
                    currency = currency_match.group(0) if currency_match else "USD"
                    
                    num_match = re.search(r'[\d.,]+', price_str)
                    value = float(num_match.group(0).replace(',', '')) if num_match else None
                    
                    unit_match = re.search(r'per\s+([^\s]+)', price_str, re.IGNORECASE)
                    unit = unit_match.group(1) if unit_match else "token"
                    
                    if 'input' in key:
                        pricing['input_cost'] = value
                    elif 'output' in key:
                        pricing['output_cost'] = value
                    else:
                        pricing['price'] = value
                    
                    pricing['currency'] = currency
                    pricing['unit'] = unit
        
        if pricing:
            item['pricing'] = pricing
        
        return item

    def _process_extracted_data(self, raw_data):
        """
        Process and clean the extracted data with enhanced normalization and deduplication.

        Args:
            raw_data (List[dict]): The raw data extracted from the source.

        Returns:
            List[dict]: The cleaned and standardized data.
        """
        processed_data = []
        
        if not isinstance(raw_data, list):
            if isinstance(raw_data, dict):
                raw_data = [raw_data]
            else:
                print("Warning: Extracted content is not a list or dict")
                return []
        
        for item in raw_data:
            if not item:
                continue
                
            model_data = dict(item)
            
            if not model_data.get('model_name'):
                continue
                
            model_data['model_name'] = model_data['model_name'].strip()
            
            if not model_data.get('benchmarks'):
                model_data['benchmarks'] = {}
            
            benchmark_fields = ['arena_score', 'mmlu', 'gsm8k', 'math', 'truthfulqa', 'humaneval']
            for field in benchmark_fields:
                if field in model_data and model_data[field]:
                    model_data['benchmarks'][field] = model_data[field]
                    del model_data[field]
            
            # Special handling for ci_95 and confidence_interval
            if 'ci_95' in model_data and model_data['ci_95']:
                model_data['benchmarks']['ci_95'] = model_data['ci_95']
                del model_data['ci_95']
            
            if 'confidence_interval' in model_data and model_data['confidence_interval']:
                model_data['benchmarks']['confidence_interval'] = model_data['confidence_interval']
                del model_data['confidence_interval']
            
            if 'organization' in model_data and not model_data.get('provider'):
                model_data['provider'] = model_data['organization']
                if 'organization' != 'provider':
                    del model_data['organization']
            
            if 'parameters' in model_data and model_data['parameters']:
                if isinstance(model_data['parameters'], str):
                    param_str = model_data['parameters'].lower()
                    if 'b' in param_str:
                        try:
                            model_data['parameters'] = int(float(param_str.replace('b', '').strip()))
                        except ValueError:
                            pass  
                    elif 'm' in param_str:
                        try:
                            model_data['parameters'] = int(float(param_str.replace('m', '').strip()) / 1000)
                        except ValueError:
                            pass 
            
            for field in ['context_window', 'max_tokens']:
                if field in model_data and model_data[field] and isinstance(model_data[field], str):
                    num_match = re.search(r'[\d,]+', model_data[field])
                    if num_match:
                        try:
                            model_data[field] = int(num_match.group(0).replace(',', ''))
                        except ValueError:
                            pass 
            
            if 'tasks' in model_data and model_data['tasks']:
                if isinstance(model_data['tasks'], str):
                    model_data['tasks'] = [t.strip() for t in re.split(r'[,;|]', model_data['tasks'])]
            
            model_data = self._extract_pricing_info(model_data)
            
            processed_data.append(model_data)
        
        merged_data = []
        seen_models = {}
        
        for item in processed_data:
            norm_name = self._normalize_model_name(item['model_name'])
            
            if norm_name in seen_models:
                existing = seen_models[norm_name]
                
                for key, value in item.items():
                    if value is not None and (key not in existing or existing[key] is None):
                        existing[key] = value
                    elif isinstance(value, dict) and isinstance(existing.get(key), dict):
                        for k, v in value.items():
                            if v is not None and (k not in existing[key] or existing[key][k] is None):
                                existing[key][k] = v
                    elif isinstance(value, list) and isinstance(existing.get(key), list):
                        # Fixed code to handle lists of dictionaries
                        if value and isinstance(value[0], dict):
                            # For lists of dictionaries, just concatenate
                            existing[key] = existing[key] + value
                        else:
                            # For lists of hashable types, try to use set for deduplication
                            try:
                                existing[key] = list(set(existing[key] + value))
                            except TypeError:  # Fallback if items are unhashable
                                combined = existing[key].copy()
                                for item in value:
                                    if item not in combined:
                                        combined.append(item)
                                existing[key] = combined
            else:
                seen_models[norm_name] = item
                merged_data.append(item)
        
        return merged_data

    async def _process_url(self, url, crawler, crawl_config):
        """
        Process a single URL to extract AI model data.
        
        Args:
            url (str): The URL to process
            crawler (AsyncWebCrawler): The configured crawler
            crawl_config (CrawlerRunConfig): The crawl configuration
            
        Returns:
            list: List of extracted AI models
        """
        print(f"\n=== Processing URL: {url} ===")
        
        try:
            print(f"Loading and interacting with the page to get content...")
            pre_crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            
            simplified_interaction = """
                // Scroll through the page to load lazy content
                for (let i = 0; i < 10; i++) {
                    window.scrollTo(0, document.body.scrollHeight * (i/5));
                    await new Promise(r => setTimeout(r, 800));
                }
                
                // Try to click on "show more" buttons
                const showMoreButtons = document.querySelectorAll('button, a');
                for (const button of showMoreButtons) {
                    if (button.textContent.toLowerCase().includes('show more') || 
                        button.textContent.toLowerCase().includes('Explore detailed pricing')) {
                        try {
                            button.click();
                            await new Promise(r => setTimeout(r, 500));
                        } catch (e) {}
                    }
                }
                
                // Return to top
                window.scrollTo(0, 0);
            """
            
            pre_crawler_run = await crawler.arun(
                url=url,
                config=pre_crawl_config,
                pre_extraction_js=simplified_interaction
            )
            
            async for pre_result in pre_crawler_run:
                pass
                
            print("Page pre-processing completed")
            
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"Warning: Pre-processing encountered an error: {str(e)}")
            print("Continuing with extraction anyway...")
            await asyncio.sleep(3)
        
        print("Starting content extraction...")
        
        extracted_content = None
        success = False
        error_message = None
        
        async def process_results():
            nonlocal extracted_content, success, error_message
            try:
                lxml_config = CrawlerRunConfig(
                    extraction_strategy=crawl_config.extraction_strategy,
                    deep_crawl_strategy=crawl_config.deep_crawl_strategy,
                    cache_mode=CacheMode.BYPASS,
                    scraping_strategy=LXMLWebScrapingStrategy(),
                    stream=True,
                    verbose=True
                )
                
                crawler_run = await crawler.arun(url=url, config=lxml_config)
                async for result in crawler_run:
                    if hasattr(result, 'extracted_content'):
                        if isinstance(result.extracted_content, list) and len(result.extracted_content) > 0:
                            if extracted_content is None:
                                extracted_content = result.extracted_content
                            else:
                                extracted_content.extend(result.extracted_content)
                        else:
                            extracted_content = result.extracted_content
                        success = True
                    if hasattr(result, 'error_message'):
                        error_message = result.error_message
                
                return success
            except Exception as e:
                error_message = str(e)
                return False
        
        await self._retry_with_timeout(
            process_results,
            retries=8,
            delay=self.retry_base_delay
        )
        
        if success and extracted_content:
            try:
                if isinstance(extracted_content, str):
                    try:
                        raw_data = json.loads(extracted_content)
                    except json.JSONDecodeError:
                        print("Warning: Extraction returned text instead of JSON. Attempting to reinterpret...")
                        wrapped_content = f'{{"model_name": "Unknown", "description": {json.dumps(extracted_content)}}}'
                        try:
                            raw_data = json.loads(wrapped_content)
                        except:
                            print("Failed to parse content as JSON.")
                            return []
                else:
                    raw_data = extracted_content
                
                processed_data = self._process_extracted_data(raw_data)
                print(f"Successfully extracted {len(processed_data)} models from {url}")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                domain = re.search(r'https?://(?:www\.)?([^/]+)', url).group(1) if re.search(r'https?://(?:www\.)?([^/]+)', url) else "unknown"
                intermediate_filename = f"intermediate_{domain}_{timestamp}.json"
                intermediate_path = os.path.join(self.output_dir, intermediate_filename)
                
                with open(intermediate_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, indent=2, ensure_ascii=False)
                    
                print(f"Saved intermediate results to {intermediate_path}")
                
                return processed_data
                
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse extracted content as JSON: {str(e)}")
                print("Raw content sample:", str(extracted_content)[:200] if extracted_content else "None")
                return []
        else:
            print(f"Error during extraction: {error_message or 'Failed after retries'}")
            return []

    async def crawl_and_extract(self):
        """
        Crawl websites and extract AI model data with improved rate limiting.

        Returns:
            str: The path to the output JSON file if extraction is successful, None otherwise.
        """
        try:
            import litellm
            litellm._turn_on_debug() 
        except ImportError:
            print("Warning: litellm not found, debug mode not enabled")
        
        llm_strategy = self._create_llm_strategy()
        
        deep_crawl_strategy = self._create_deep_crawl_strategy()
        
        crawl_config = CrawlerRunConfig(
            extraction_strategy=llm_strategy,
            deep_crawl_strategy=deep_crawl_strategy,
            cache_mode=CacheMode.BYPASS,
            scraping_strategy=LXMLWebScrapingStrategy(),
            stream=True,
            verbose=True
        )
        
        browser_cfg = BrowserConfig(
            headless=True, 
            verbose=True,
        )
        
        all_models = []
        url_results = {}  
        
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            for i, url in enumerate(self.urls):
                if i > 0:
                    delay = self.delay_between_urls + random.uniform(0, 10) 
                    print(f"\nWaiting {delay:.1f} seconds before processing next URL to avoid rate limits...")
                    await asyncio.sleep(delay)
                
                models = await self._process_url(url, crawler, crawl_config)
                url_results[url] = len(models) if models else 0
                
                if models:
                    all_models.extend(models)
                    print(f"Added {len(models)} models from {url} to the collection")
                else:
                    print(f"No models were extracted from {url}")
            
        if all_models:
            diag_filename = f"crawl_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            diag_path = os.path.join(self.output_dir, diag_filename)
            with open(diag_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "crawl_time": datetime.now().isoformat(),
                    "total_models_extracted": len(all_models),
                    "urls_processed": len(self.urls),
                    "results_by_url": url_results
                }, f, indent=2)
            
            save_to_database(all_models)  
            output_file_json = self.save_to_json(all_models) 
            print(f"\nCrawling completed! Successfully extracted {len(all_models)} AI models from {len(self.urls)} URLs.")
            print(f"Extraction summary by URL: {url_results}")
            
            return output_file_json  
        else:
            print("Error: No AI models were extracted.")
            return None

async def main():
    """
    Main function that creates a crawler instance and runs the extraction process.

    Returns:
        int: 0 if extraction was successful, 1 otherwise.
    """
    URLS_TO_SCRAPE = [
   #"https://openai.com/api/pricing/",
   # "https://groq.com/pricing/",
    "https://web.lmarena.ai/leaderboard",
    "https://www.vellum.ai/llm-leaderboard"
    ]
    
    crawler = AIModelCrawler(
        urls=URLS_TO_SCRAPE,
        api_token=os.getenv('GROQ_API_TOKENS'),
        output_dir="./output",
        delay_between_urls=12,  
        retry_base_delay=20    
    )
    
    result = await crawler.crawl_and_extract()
    
    return 0 if result else 1

if __name__ == "__main__":
    asyncio.run(main())