These files were run in kaggle notebook.


Each file was placed in a different cell with " %%writefile name_of_file " ritten at top of each cell.

Example :

![Screenshot 2025-05-01 134506](https://github.com/user-attachments/assets/3ab078a1-6dc3-469a-a051-7b465b588f8a)


Command used to run the files :

'''
!python langchain_main.py --input "/kaggle/input/wtsp-chat/WhatsApp Chat with Murugan dada group .txt" --output "/kaggle/working/new_output13.json" --model "mistralai/Mistral-7B-Instruct-v0.3" --batch-size 2 --quantize none   --use-bettertransformer --timeout 600 --offload-folder "/kaggle/working/new_offload" --limit 8 --no-use-vllm --max-new-tokens 512  --device "auto" 
'''
