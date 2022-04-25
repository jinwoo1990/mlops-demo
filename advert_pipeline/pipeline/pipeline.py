# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, List, Optional

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    # query: str,  # either data_path or query can be used for ingesting data  # TODO: query로 변경
    preprocessing_fn: str,
    run_fn: str,
    train_args: tfx.proto.TrainArgs,
    eval_args: tfx.proto.EvalArgs,
    eval_config: str,
    serving_model_dir: str,
    schema_path: Optional[str] = None,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[str]] = None
) -> tfx.dsl.Pipeline:
    components = []

    # Ingests data.
    example_gen = tfx.components.CsvExampleGen(input_base=data_path)  # Csv
    # example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
    #     query=query)  # BigQuery
    components.append(example_gen)

    # Computes statistics.
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])
    components.append(statistics_gen)

    # Generates schema.
    if schema_path is None:
        schema_gen = tfx.components.SchemaGen(
            statistics=statistics_gen.outputs['statistics'])
        components.append(schema_gen)
    else:
        # Import user provided schema into the pipeline.
        schema_gen = tfx.components.ImportSchemaGen(schema_file=schema_path)
        components.append(schema_gen)

    # Performs anomaly detection.
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    components.append(example_validator)

    # Performs transformations.
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        preprocessing_fn=preprocessing_fn)
    components.append(transform)

    # Train a model.
    trainer_args = {
        'run_fn': run_fn,
        'examples': transform.outputs['transformed_examples'],
        'schema': schema_gen.outputs['schema'],
        'transform_graph': transform.outputs['transform_graph'],
        'train_args': train_args,
        'eval_args': eval_args,
    }
    trainer = tfx.components.Trainer(**trainer_args)
    components.append(trainer)

    # Defines resolver to find latest blessed model 
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
        'latest_blessed_model_resolver')
    components.append(model_resolver)

    # Perform quality validation of a candidate model (compared to a baseline usng TMFA).
    eval_config = text_format.Parse(eval_config, tfma.EvalConfig())
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],  # resolver로 baseline 모델 찾기
        eval_config=eval_config)
    components.append(evaluator)

    # Push model which passes evaluation criteria.
    pusher_args = {
        'model':
            trainer.outputs['model'],
        'model_blessing':
            evaluator.outputs['blessing'],  # BLESSED로 나와야 통과
        'push_destination':
            tfx.proto.PushDestination(
                filesystem=tfx.proto.PushDestination.Filesystem(
                    base_directory=serving_model_dir))
    }
    pusher = tfx.components.Pusher(**pusher_args)
    components.append(pusher)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        # enable_cache=True,  # default=False
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )
