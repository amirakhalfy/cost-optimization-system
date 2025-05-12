from datetime import datetime
from app.db.models import Provider, Model, Pricing, Benchmark, Task, ModelTask, BenchmarkResult
from app.db.db_setup import SessionLocal

def save_to_database(all_models):
    """
    Save extracted AI model data to the database.

    Args:
        all_models (list): List of dictionaries containing model data.

    Returns:
        str: A success message if data is saved, None if an error occurs.
    """
    session = SessionLocal()
    
    try:
        for model_data in all_models:
            provider_name = model_data.get('provider')
            if not provider_name:
                provider_name = "Unknown Provider"
                
            provider = session.query(Provider).filter_by(name=provider_name).first()
            if not provider:
                provider = Provider(name=provider_name, created_at=datetime.now())
                session.add(provider)
                session.commit()
            provider_id = provider.id

            new_model = Model(
                name=model_data['model_name'],
                provider_id=provider_id,
                license=model_data.get('license'),
                description=model_data.get('description'),
                context_window=model_data.get('context_window'),
                max_tokens=model_data.get('max_tokens'),
                parameters=model_data.get('parameters'),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            session.add(new_model)
            session.commit()
            model_id = new_model.id

            if model_data.get('pricing') and isinstance(model_data['pricing'], dict):
                pricing_dict = model_data['pricing']
                
                token_unit = pricing_dict.get('unit')
                if token_unit is None:
                    token_unit = 1000000  
                
                if isinstance(token_unit, str):
                    if 'million' in token_unit.lower() or 'M' in token_unit:
                        token_unit = 1000000
                    elif 'thousand' in token_unit.lower() or 'K' in token_unit:
                        token_unit = 1000
                    else:
                        token_unit = 1
                
                currency = pricing_dict.get('currency')
                if currency is None:
                    currency = 'USD'  
                
                pricing = Pricing(
                    model_id=model_id,
                    input_cost=float(pricing_dict.get('input_cost', 0.0) or 0.0),
                    output_cost=float(pricing_dict.get('output_cost', 0.0) or 0.0),
                    cached_input=pricing_dict.get('cached_input'),
                    training_cost=pricing_dict.get('training_cost'),
                    token_unit=token_unit, 
                    currency=currency, 
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                session.add(pricing)

            if model_data.get('benchmarks') and isinstance(model_data['benchmarks'], dict):
                benchmarks_dict = model_data['benchmarks']
                if any(key in benchmarks_dict for key in ['arena_score', 'confidence_interval', 'votes']):
                    benchmark = Benchmark(
                        name="arena",  
                        score_benchmark=benchmarks_dict.get('arena_score', 0.0) or 0.0, 
                        arena_score=benchmarks_dict.get('arena_score'),
                        confidence_interval=benchmarks_dict.get('confidence_interval'),
                        votes=benchmarks_dict.get('votes'),
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    session.add(benchmark)
                    session.commit()
                    
                    result = BenchmarkResult(
                        model_id=model_id,
                        benchmark_id=benchmark.id,
                        score=float(benchmarks_dict.get('arena_score', 0.0) or 0.0),  
                        created_at=datetime.now()
                    )
                    session.add(result)
                else:
                    for name, score in benchmarks_dict.items():
                        if not name or (score is None and not isinstance(score, (int, float))):
                            continue
                            
                        try:
                            score_float = float(score) if score is not None else 0.0
                        except (ValueError, TypeError):
                            score_float = 0.0
                            
                        benchmark = Benchmark(
                            name=name,
                            score_benchmark=score_float,
                            created_at=datetime.now(),
                            updated_at=datetime.now()
                        )
                        session.add(benchmark)
                        session.commit()
                        
                        result = BenchmarkResult(
                            model_id=model_id,
                            benchmark_id=benchmark.id,
                            score=score_float, 
                            created_at=datetime.now()
                        )
                        session.add(result)

            tasks_list = model_data.get('tasks')
            if tasks_list:
                if isinstance(tasks_list, str):
                    tasks_list = [tasks_list]
                elif not isinstance(tasks_list, list):
                    tasks_list = []

                for task_name in tasks_list:
                    if task_name:  
                        task = session.query(Task).filter_by(task_name=task_name).first()
                        if not task:
                            task = Task(
                                task_name=task_name,
                                task_category='unknown', 
                                created_at=datetime.now(),
                                updated_at=datetime.now()
                            )
                            session.add(task)
                            session.commit()
                        model_task = ModelTask(model_id=model_id, task_id=task.id)
                        session.add(model_task)

        session.commit()
        success_message = f"Successfully saved {len(all_models)} models to the database."
        print(success_message)
        return success_message

    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {e}")
        return None

    finally:
        session.close()