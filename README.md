# llama.ttf

A font containing a large language model and inference engine.

## What?

A font containing a large language model and inference engine.

## Why?

## What?

## Usage

Just download [`llama.ttf`](https://github.com/fuglede/llama.ttf/raw/master/llamattf/llama.ttf) and use it like you would any other font, for instance by adding it to `~/.fonts`. Then, use it somewhere where [Harfbuzz](https://github.com/harfbuzz/harfbuzz) is used and built with Wasm support.

The simplest way to do that with this is probably to build [Harfbuzz](https://github.com/harfbuzz/harfbuzz/tree/4cfc6d8e173e800df086d7be078da2e8c5cfca19) with `-Dwasm=enabled`, and build [wasm-micro-runtime](https://github.com/bytecodealliance/wasm-micro-runtime/tree/382d52fc05dbb543dfafb969182104d6c4856c63), then add the resulting shared libraries, `libharfbuzz.so.0.60811.0` and `libiwasm.so` to `LD_PRELOAD` before running a Harfbuzz-based application such as gedit or GIMP; no recompilation of the applications is required.

## Demo and more info

See [https://fuglede.github.io/llama.ttf/](https://fuglede.github.io/llama.ttf/).
