from openai import OpenAI
import os
import pandas as pd
from tqdm import tqdm
import time
import sys
from typing import Any, Callable, Optional

sys.path.append("../")


class LLM_wrapper:
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        output_format_class: Any,
    ) -> None:
        """
        Initialize the wrapper for structured LLM generation.

        :param model_name: Name of the OpenAI model to use for generation.
        :type model_name: str
        :param system_prompt: System prompt prepended to every model call.
        :type system_prompt: str
        :param output_format_class: Structured output schema used to parse responses.
        :type output_format_class: Any
        """
        # Initialize OpenAI client for structured response generation
        self.client = OpenAI()
        # Store the model name used for inference (e.g., "gpt-5.2-2025-12-11")
        self.model_name = model_name
        # Store the system prompt that will be prepended to each request
        self.system_prompt = system_prompt
        # Store the Pydantic model (or equivalent) used for structured parsing
        self.output_format_class = output_format_class

    def generate_nan_output(self) -> Any:
        """
        Generate an empty structured output instance.

        Each field defined in ``output_format_class`` is filled with ``None``.

        :return: Empty instance of the structured output schema.
        :rtype: Any
        """
        # Get all expected output field names from the structured output schema
        fields = self.output_format_class.model_fields.keys()
        # Create a dictionary with None for every expected field
        empty_values = {field: None for field in fields}
        # Return an empty instance of the output schema
        return self.output_format_class(**empty_values)

    def assess_one_row(self, user_prompt: str, max_retries: int = 3) -> Any:
        """
        Run structured generation for a single prompt.

        :param user_prompt: User prompt to send to the model.
        :type user_prompt: str
        :param max_retries: Maximum number of attempts for the API call.
        :type max_retries: int
        :return: Parsed structured output for the prompt, or an empty output object
            if all retries fail.
        :rtype: Any
        """
        # Retry the LLM call up to max_retries times
        for attempt in range(max_retries):
            try:
                # Call the Responses API and request structured parsing
                resp = self.client.responses.parse(
                    model=self.model_name,
                    input=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text_format=self.output_format_class,
                    truncation="auto",
                )
                # Return the parsed structured output
                return resp.output_parsed

            except Exception as e:
                # Print any API or parsing error encountered during generation
                print(f"Error during LLM call: {e}")
                # If all retries are exhausted, return a null-valued output object
                if attempt == max_retries:
                    return self.generate_nan_output()
                # Backoff slightly before retrying
                time.sleep(0.6 * attempt)

    @staticmethod
    def load_input_file(
        input_file_path: str,
        text_column: str,
        debug_mode: bool,
    ) -> pd.DataFrame:
        """
        Load an input file and validate the required text column.

        Supports CSV, Parquet, and Brotli-compressed Parquet files.

        :param input_file_path: Path to the input file.
        :type input_file_path: str
        :param text_column: Name of the column containing prompt text.
        :type text_column: str
        :param debug_mode: Whether to restrict the returned dataframe to the first
            10 rows.
        :type debug_mode: bool
        :raises ValueError: If ``text_column`` is missing from the input file.
        :return: Loaded dataframe, optionally truncated in debug mode.
        :rtype: pd.DataFrame
        """
        # Load CSV input
        if input_file_path.endswith(".csv"):
            df = pd.read_csv(input_file_path)
        # Load parquet or brotli-compressed parquet input
        elif input_file_path.endswith(".parquet.br") or input_file_path.endswith(
            ".parquet"
        ):
            df = pd.read_parquet(input_file_path)
        # Ensure the text column required for prompting is present
        if text_column not in df.columns:
            raise ValueError(
                f"Missing required column '{text_column}' in the input file"
            )
        # In debug mode, restrict processing to the first 10 rows
        if debug_mode:
            return df.head(10)
        # Otherwise return the full dataframe
        return df

    def assess_dataframe(
        self,
        input_file_path: str,
        text_column: str,
        output_dir: str,
        save_ckpt_every: int = 10,
        debug_mode: bool = False,
        row_filter: Optional[Callable[[pd.Series], bool]] = None,
    ) -> pd.DataFrame:
        """
        Run structured generation over a dataframe and save periodic checkpoints.

        If a checkpoint already exists for the same model, the latest one is loaded
        and processing resumes from unprocessed rows only.

        :param input_file_path: Path to the input file.
        :type input_file_path: str
        :param text_column: Name of the column containing prompt text.
        :type text_column: str
        :param output_dir: Directory where checkpoints and outputs are stored.
        :type output_dir: str
        :param save_ckpt_every: Number of processed rows between checkpoint saves.
        :type save_ckpt_every: int
        :param debug_mode: Whether to restrict input loading to the first 10 rows.
        :type debug_mode: bool
        :param row_filter: Optional predicate applied row-wise to restrict which
            unprocessed rows are assessed.
        :type row_filter: Optional[Callable[[pd.Series], bool]]
        :return: Dataframe containing the generated outputs.
        :rtype: pd.DataFrame
        """
        # Create checkpoint directory inside the output directory
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create timestamped filenames for checkpoint and final output
        ts = time.strftime("%Y%m%d_%H%M")
        checkpoint_filename = f"{ts}_{self.model_name}.checkpoint.csv"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        # load latest checkpoint for this model if present (any timestamp for same model)
        pattern = f"_{self.model_name}.checkpoint.csv"
        candidates = [
            os.path.join(checkpoint_dir, fn)
            for fn in os.listdir(checkpoint_dir)
            if fn.endswith(pattern)
        ]
        if candidates:
            # Resume from the most recently modified checkpoint
            latest_ckpt = max(candidates, key=os.path.getmtime)
            print(f"Loading checkpoint from {latest_ckpt}")
            df = pd.read_csv(latest_ckpt)
            checkpoint_path = latest_ckpt

        else:
            # Otherwise load the original input file
            df = self.load_input_file(
                input_file_path=input_file_path,
                text_column=text_column,
                debug_mode=debug_mode,
            )

        # ensure generation column exists
        if "generation" not in df.columns:
            # Track whether each row has already been processed
            df["generation"] = pd.NA

        # ensure all model output columns exist
        for field in self.output_format_class.model_fields.keys():
            if field not in df.columns:
                # Pre-create output columns for structured fields
                df[field] = pd.NA

        # Work only on rows without a generation yet
        unprocessed_df = df[df["generation"].isna()]

        if row_filter is not None:
            # Apply optional row-level filtering only to unprocessed rows
            unprocessed_df = unprocessed_df[unprocessed_df.apply(row_filter, axis=1)]

        # Compute summary counts for logging
        n_total = len(df)
        n_unprocessed = df["generation"].isna().sum()
        n_filtered = len(unprocessed_df)

        print(
            f"Total rows: {n_total}, "
            f"Already processed: {n_total - n_unprocessed}, "
            f"Remaining (after filter): {n_filtered}"
        )

        # Iterate over rows still needing processing
        for i, (idx, row) in enumerate(
            tqdm(
                unprocessed_df.iterrows(),
                total=len(unprocessed_df),
            )
        ):
            # Extract the text content that will be sent to the model
            input_text = row[text_column]
            try:
                # Run structured assessment for a single row
                assessment = self.assess_one_row(user_prompt=input_text)
                # write assessment fields into df
                for k, v in assessment.model_dump().items():
                    # Store each parsed field back into the original dataframe
                    df.at[idx, k] = v
                # Mark row as successfully generated
                df.at[idx, "generation"] = True

            except Exception as e:
                # Log row-level failures without stopping the whole run
                print(f"Error with row {idx}: {e}")
                df.at[idx, "generation"] = pd.NA

            # Save checkpoint periodically
            if (i + 1) % save_ckpt_every == 0 or i == len(unprocessed_df) - 1:
                df.to_csv(checkpoint_path, index=False)

        # Final save
        print("Processing complete. Final results saved.")
        return df
