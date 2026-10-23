# NspireLM build. MODEL names a directory produced by compiler/nslmc.py.
#   make MODEL=build/stories15M host        native reference build
#   make MODEL=build/stories15M arm         ARMv5TE (arm926ej-s) Linux build for qemu-arm
MODEL ?= build/stories260K
CC ?= cc
ARMCC ?= arm-linux-gnueabi-gcc
CFLAGS ?= -O2
WARN = -std=c11 -Wall -Wextra -Werror -Wno-unused-function
SRC = runtime/nslm.c runtime/nslm_kern.c runtime/nslm_tok.c host/main.c
INC = -Iruntime -I$(MODEL)

host: $(MODEL)/nslm_host
arm: $(MODEL)/nslm_arm

$(MODEL)/nslm_host: $(SRC) runtime/nslm.h $(MODEL)/nslm_model.h
	$(CC) $(CFLAGS) $(WARN) -D_POSIX_C_SOURCE=199309L $(INC) $(SRC) -o $@

$(MODEL)/nslm_arm: $(SRC) runtime/nslm.h $(MODEL)/nslm_model.h
	$(ARMCC) $(CFLAGS) $(WARN) -D_POSIX_C_SOURCE=199309L -mcpu=arm926ej-s -marm -static $(INC) $(SRC) -o $@

.PHONY: host arm
