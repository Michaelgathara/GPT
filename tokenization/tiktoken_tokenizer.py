def get_tiktoken_tokenizer(model_name="gpt-2"):
    import tiktoken
    encoding = tiktoken.encoding_for_model(model_name)
    return encoding

