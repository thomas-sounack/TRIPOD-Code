from environs import Env
from openai import OpenAI
import os
import pandas as pd
from tqdm import tqdm
import time
import sys

sys.path.append("../")


# from dotenv import load_dotenv

# load_dotenv()


class LLM_wrapper:
    def __init__(self, model_name: str, system_prompt: str, output_format_class):
        self.client = OpenAI()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.output_format_class = output_format_class

    def generate_nan_output(self):
        fields = self.output_format_class.model_fields.keys()
        empty_values = {field: None for field in fields}
        return self.output_format_class(**empty_values)

    def assess_one_row(self, user_prompt: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                resp = self.client.responses.parse(
                    model=self.model_name,
                    input=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text_format=self.output_format_class,
                )
                return resp.output_parsed

            except Exception:
                if attempt == max_retries:
                    return self.generate_nan_output()
                time.sleep(0.6 * attempt)

    @staticmethod
    def load_input_file(input_file_path: str, text_column: str, debug_mode: bool):
        df = pd.read_csv(input_file_path)
        if text_column not in df.columns:
            raise ValueError(
                f"Missing required column '{text_column}' in the input file"
            )
        if debug_mode:
            return df.head(10)
        return df

    def assess_dataframe(
        self,
        input_file_path: str,
        text_column: str,
        output_dir: str,
        save_ckpt_every: int = 10,
        debug_mode: bool = False,
    ):
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M")
        checkpoint_filename = f"{ts}_{self.model_name}.checkpoint.csv"
        output_filename = f"{ts}_{self.model_name}.csv"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        # load latest checkpoint for this model if present (any timestamp for same model)
        pattern = f"_{self.model_name}.checkpoint.csv"
        candidates = [
            os.path.join(checkpoint_dir, fn)
            for fn in os.listdir(checkpoint_dir)
            if fn.endswith(pattern)
        ]
        if candidates:
            latest_ckpt = max(candidates, key=os.path.getmtime)
            print(f"Loading checkpoint from {latest_ckpt}")
            df = pd.read_csv(latest_ckpt)
            checkpoint_path = latest_ckpt

        else:
            df = self.load_input_file(
                input_file_path=input_file_path,
                text_column=text_column,
                debug_mode=debug_mode,
            )

        # ensure generation column exists
        if "generation" not in df.columns:
            df["generation"] = pd.NA

        # ensure all model output columns exist
        for field in self.output_format_class.model_fields.keys():
            if field not in df.columns:
                df[field] = pd.NA

        # Work only on rows without a generation yet
        unprocessed_df = df[df["generation"].isna()]

        print(
            f"Total rows: {len(df)}, Already processed: {len(df) - len(unprocessed_df)}, Remaining: {len(unprocessed_df)}"
        )

        for i, (idx, row) in enumerate(
            tqdm(
                unprocessed_df.iterrows(),
                total=len(unprocessed_df),
            )
        ):
            input_text = row[text_column]
            try:
                assessment = self.assess_one_row(user_prompt=input_text)
                # write assessment fields into df
                for k, v in assessment.model_dump().items():
                    df.at[idx, k] = v
                df.at[idx, "generation"] = True

            except Exception as e:
                print(f"Error with row {idx}: {e}")
                df.at[idx, "generation"] = pd.NA

            # Save checkpoint periodically
            if (i + 1) % save_ckpt_every == 0 or i == len(unprocessed_df) - 1:
                df.to_csv(checkpoint_path, index=False)

        # Final save
        df.to_csv(output_filename, index=False)
        print("Processing complete. Final results saved.")
