These files were run in kaggle notebook.


Each source file was written to a separate Kaggle notebook cell using the `%%writefile` magic:

<pre>
%%writefile langchain_extractor.py
# Your Python code goes here...
</pre>

Example :

![Screenshot 2025-05-01 134506](https://github.com/user-attachments/assets/3ab078a1-6dc3-469a-a051-7b465b588f8a)


Command used to run the files :

<pre>
!python langchain_main.py --input "/kaggle/input/wtsp-chat/WhatsApp Chat with Murugan dada group .txt" --output "/kaggle/working/new_output13.json" --model "mistralai/Mistral-7B-Instruct-v0.3" --batch-size 2 --quantize none   --use-bettertransformer --timeout 600 --offload-folder "/kaggle/working/new_offload" --limit 8 --no-use-vllm --max-new-tokens 512  --device "auto" 
</pre>

Model was loaded using hugging face :

Authenticate with Hugging Face using your Kaggle secrets:

<pre>
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

# Load HF token from Kaggle secrets
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("hf")

# Login to Hugging Face
login(token=hf_token)
</pre>

Save your token in Kaggle → Secrets with the key name: hf

