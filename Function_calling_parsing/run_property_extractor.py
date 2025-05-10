import asyncio
from property_extractor_function_calling import PropertyExtractorFunctionCalling

if __name__ == "__main__":
    print("Starting property extraction...")  # Add this line
    tokenizer_path = "/kaggle/working/mistral_models/7B-Instruct-v0.3/tokenizer.model.v3"
    model_path = "/kaggle/working/mistral_models/7B-Instruct-v0.3"
    input_file = "/kaggle/input/your_input_file.txt"  # Place your input file in the Kaggle dataset
    output_file = "/kaggle/working/your_output_file.json"

    extractor = PropertyExtractorFunctionCalling(tokenizer_path, model_path, batch_size=2, max_new_tokens=256)
    asyncio.run(extractor.process_file_async(input_file, output_file))
    print("Extraction finished.")  # Add this line