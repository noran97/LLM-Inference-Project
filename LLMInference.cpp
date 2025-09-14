#include "../LLMInference.h"
#include <cstring>
#include <iostream>
#include <sstream>

void LLMInference::loadModel(const std::string& model_path, float min_p, float temperature) {
    // create an instance of llama_model
    llama_model_params model_params = llama_model_default_params();
    _model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (!_model) {
        throw std::runtime_error("load_model() failed");
    }

    // create an instance of llama_context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;           
    ctx_params.no_perf = true;         // disable performance metrics
    _ctx = llama_init_from_model(_model, ctx_params);

    if (!_ctx) {
        throw std::runtime_error("llama_init_from_model() returned null");
    }

    // initialize sampler
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;     // disable performance metrics
    _sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(_sampler, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    _formattedMessages = std::vector<char>(4096);
    _messages.clear();

    // initialize counters
    _prevLen  = 0;
    _nCtxUsed = 0;
    _response.clear();
}

void LLMInference::startCompletion(const std::string& query) {
    addChatMessage(query, "user");

    // Build a simple prompt format - works with most models
    std::string prompt = "";
    for (const auto& message : _messages) {
        if (strcmp(message.role, "system") == 0) {
            prompt += "System: " + std::string(message.content) + "\n\n";
        } else if (strcmp(message.role, "user") == 0) {
            prompt += "User: " + std::string(message.content) + "\n\n";
        } else if (strcmp(message.role, "assistant") == 0) {
            prompt += "Assistant: " + std::string(message.content) + "\n\n";
        }
    }
    prompt += "Assistant: ";

    std::cout << "DEBUG: Full prompt:\n" << prompt << std::endl;

    // Tokenization
    _promptTokens = common_tokenize(_ctx, prompt, false, true);
    
    std::cout << "DEBUG: Tokenized " << _promptTokens.size() << " tokens" << std::endl;

    // Correctly create a llama_batch manually, as llama_batch_add is not available.
    _batch = llama_batch_init(_promptTokens.size(), 0, 1);
    
    // Store the current context size before the new batch is added
    int current_context_size = _nCtxUsed;

    for (int i = 0; i < (int)_promptTokens.size(); i++) {
        _batch.token[i] = _promptTokens[i];
        // FIX: The position must be offset by the number of tokens already in the KV cache
        _batch.pos[i] = current_context_size + i;
        // Allocate and set the sequence ID correctly for each token.
        _batch.seq_id[i] = (llama_seq_id*)malloc(sizeof(llama_seq_id) * 1);
        _batch.seq_id[i][0] = 0;
        _batch.n_seq_id[i] = 1;
        // Only set logits to true for the final token in the prompt
        _batch.logits[i] = (i == (int)_promptTokens.size() - 1);
    }
    // The total number of tokens in the batch
    _batch.n_tokens = _promptTokens.size();

    // update the context-used counter
    _nCtxUsed = current_context_size + _promptTokens.size();
    
    std::cout << "DEBUG: Context used: " << _nCtxUsed << std::endl;
    
    // Clear any previous response
    _response.clear();
}

std::string LLMInference::completionLoop() {
    // context size
    int contextSize = llama_n_ctx(_ctx);

    // check context size before decoding
    if (_nCtxUsed >= contextSize - 1) {
        std::cerr << "context size exceeded" << '\n';
        addChatMessage(_response, "assistant"); // Add the incomplete response to history
        return "[EOG]";
    }

    // run the model (decode the current batch)
    if (llama_decode(_ctx, _batch) < 0) {
        throw std::runtime_error("llama_decode() failed");
    }

    // sample a token
    _currToken = llama_sampler_sample(_sampler, _ctx, -1);

    // Check for end-of-generation token
    if (llama_vocab_is_eog(llama_model_get_vocab(_model), _currToken)) {
        addChatMessage(_response, "assistant");
        return "[EOG]";
    }

    // convert token -> piece and accumulate
    std::string piece = common_token_to_piece(_ctx, _currToken, true);
    _response += piece;

    // Free the old batch
    llama_batch_free(_batch);
    // Create a new one with a single token, manually populated
    _batch = llama_batch_init(1, 0, 1);
    
    _batch.token[0] = _currToken;
    _batch.pos[0] = (int)_nCtxUsed;
    // Allocate and set the sequence ID correctly for the single token.
    _batch.seq_id[0] = (llama_seq_id*)malloc(sizeof(llama_seq_id) * 1);
    _batch.seq_id[0][0] = 0;
    _batch.n_seq_id[0] = 1;
    _batch.logits[0] = true;
    _batch.n_tokens = 1;

    // increment context-used counter AFTER using the token
    _nCtxUsed += 1;

    return piece;
}

void LLMInference::stopCompletion() {
    if (_batch.n_tokens > 0) {
        llama_batch_free(_batch);
        _batch = {0};
    }
}

LLMInference::~LLMInference() {
    // free message text memory (allocated with strdup)
    for (llama_chat_message &message: _messages) {
        free((void*)message.content);
        free((void*)message.role);
    }

    // Free the batch. This will handle the freeing of `seq_id` arrays.
    if (_batch.n_tokens > 0) {
        llama_batch_free(_batch);
    }

    // free resources
    if (_sampler) llama_sampler_free(_sampler);
    if (_ctx) llama_free(_ctx);
    if (_model) llama_model_free(_model);
}

void LLMInference::addChatMessage(const std::string &content, const std::string &role) {
    llama_chat_message msg;
    msg.role    = strdup(role.c_str());   // allocate role string
    msg.content = strdup(content.c_str()); // allocate content string
    _messages.push_back(msg);
}
