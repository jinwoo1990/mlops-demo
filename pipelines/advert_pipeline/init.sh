#!/usr/bin/env bash

# .env 파일에서 환경변수 불러오기
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi