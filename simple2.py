# This is a port of the C++ simple.cpp example from the llama.cpp
# repository to Python. Original C++ version:
#
# https://github.com/ggerganov/llama.cpp/tree/master/examples/simple
#
# Changes from the C++ version:
#
# - Only the '-m' command line option is implemented, which is used to
#   specify the location of the model to use
#
# - This Python version introduces a function to print a token,
#   'print_token'; the C++ version does not contain such a function and
#   the code is duplicated instead

import argparse
import ctypes
import llama_cpp
import sys


def print_token(vocab, token):
    buf = (ctypes.c_char * 128)()
    n = llama_cpp.llama_token_to_piece(
        vocab, llama_cpp.llama_token(token), buf, len(buf), 0, True
    )
    if n < 0:
        print("error: failed to convert token to piece", file=sys.stderr)
        sys.exit(1)
    print(
        '"{}"'.format(buf[:n].decode("utf-8")),  # type: ignore
        flush=True,
    )


def no_log(level, text, user_data):
    pass  # do nothing


# parse command line arguments

parser = argparse.ArgumentParser()
parser.add_argument("-m", type=str, required=True)
args = parser.parse_args()

# prompt to generate text from
prompt = "Hello my name is"
# number of layers to offload to the GPU
ngl = 99
# number of tokens to predict
n_predict = 32

# disable logging

log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)(
    no_log
)
llama_cpp.llama_log_set(log_callback, ctypes.c_void_p())

# initialize backends

llama_cpp.llama_backend_init()

# initialize the model

model_params = llama_cpp.llama_model_default_params()
model_params.n_gpu_layers = ngl

model = llama_cpp.llama_load_model_from_file(args.m.encode("utf-8"), model_params)

if model is None:
    print("error: unable to load model", file=sys.stderr)
    sys.exit(1)

vocab = llama_cpp.llama_model_get_vocab(model)

if vocab is None:
    print("error: llama_model_get_vocab() failed", file=sys.stderr)
    sys.exit(1)

# tokenize the prompt

prompt_bytes = bytes(str(prompt), "utf-8")

# find the number of tokens in the prompt
n_prompt = -llama_cpp.llama_tokenize(
    vocab,
    prompt_bytes,
    len(prompt_bytes),
    None,  # type: ignore
    0,
    True,
    True,
)

# allocate space for the tokens and tokenize the prompt
prompt_tokens = (llama_cpp.llama_token * n_prompt)()
if (
    llama_cpp.llama_tokenize(
        vocab,
        prompt_bytes,
        len(prompt_bytes),
        prompt_tokens,
        len(prompt_tokens),
        True,
        True,
    )
    < 0
):
    print("error: failed to tokenize the prompt", file=sys.stderr)
    sys.exit(1)

# initialize the context

ctx_params = llama_cpp.llama_context_default_params()
# n_ctx is the context size
ctx_params.n_ctx = n_prompt + n_predict - 1
# n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
ctx_params.n_batch = n_prompt
# enable performance counters
ctx_params.no_perf = False

ctx = llama_cpp.llama_init_from_model(model, ctx_params)

if ctx is None:
    print("error: failed to create the llama_context", file=sys.stderr)
    sys.exit(1)

# initialize the sampler

sparams = llama_cpp.llama_sampler_chain_default_params()
sparams.no_perf = False
smpl = llama_cpp.llama_sampler_chain_init(sparams)

llama_cpp.llama_sampler_chain_add(smpl, llama_cpp.llama_sampler_init_greedy())

# print the prompt token-by-token

print("Prompt\n======\n")

for id in prompt_tokens:
    print_token(vocab, id)

# prepare a batch for the prompt

batch = llama_cpp.llama_batch_get_one(prompt_tokens, len(prompt_tokens))

# main loop

print("\nPredictions\n===========\n")

t_main_start = llama_cpp.llama_time_us()
n_decode = 0
new_token_id_arr = (llama_cpp.llama_token * 1)()

n_pos = 0
while n_pos + batch.n_tokens < n_prompt + n_predict:
    # evaluate the current batch with the transformer model
    if llama_cpp.llama_decode(ctx, batch) != 0:
        print("failed to eval", file=sys.stderr)
        sys.exit(1)

    n_pos += batch.n_tokens

    # sample the next token
    new_token_id_arr[0] = llama_cpp.llama_sampler_sample(smpl, ctx, -1)

    # is it an end of generation?
    if llama_cpp.llama_vocab_is_eog(vocab, new_token_id_arr[0]):
        break

    print_token(vocab, new_token_id_arr[0])

    # prepare the next batch with the sampled token
    batch = llama_cpp.llama_batch_get_one(new_token_id_arr, 1)

    n_decode += 1

print()
print()

t_main_end = llama_cpp.llama_time_us()

print(
    "decoded {} tokens in {:.2f} s, speed: {:.2f} t/s".format(
        n_decode,
        (t_main_end - t_main_start) / 1000000.0,
        n_decode / ((t_main_end - t_main_start) / 1000000.0),
    ),
    file=sys.stderr,
)

print(file=sys.stderr)
llama_cpp.llama_perf_sampler_print(smpl)
llama_cpp.llama_perf_context_print(ctx)
print(file=sys.stderr)

llama_cpp.llama_sampler_free(smpl)
llama_cpp.llama_free(ctx)
llama_cpp.llama_model_free(model)
