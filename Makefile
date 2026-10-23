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

ASM = runtime/arm/matmul_q8.S

host: $(MODEL)/nslm_host
arm: $(MODEL)/nslm_arm $(MODEL)/nslm_armc

$(MODEL)/nslm_host: $(SRC) runtime/nslm.h $(MODEL)/nslm_model.h
	$(CC) $(CFLAGS) $(WARN) -D_POSIX_C_SOURCE=200112L $(INC) $(SRC) -o $@

# ARMv5TE with the assembly kernels (what runs on the calculator)
$(MODEL)/nslm_arm: $(SRC) $(ASM) runtime/nslm.h $(MODEL)/nslm_model.h
	$(ARMCC) $(CFLAGS) $(WARN) -D_POSIX_C_SOURCE=200112L -DNSLM_HAVE_ASM_MATMUL -mcpu=arm926ej-s -marm -static $(INC) $(SRC) $(ASM) -o $@

# ARMv5TE, portable C kernels only (reference for the cycle model)
$(MODEL)/nslm_armc: $(SRC) runtime/nslm.h $(MODEL)/nslm_model.h
	$(ARMCC) $(CFLAGS) $(WARN) -D_POSIX_C_SOURCE=200112L -mcpu=arm926ej-s -marm -static $(INC) $(SRC) -o $@

APP = app/app.c app/font8x8.c
RT = runtime/nslm.c runtime/nslm_kern.c runtime/nslm_tok.c

# Headless app (scripted keys, frames to PPM) and the expected self-test hash it bakes in
app-host: $(MODEL)/nslm_app
$(MODEL)/nslm_app: $(RT) $(APP) app/plat_host.c app/plat.h runtime/nslm.h $(MODEL)/nslm_model.h
	$(CC) $(CFLAGS) $(WARN) -D_POSIX_C_SOURCE=200112L $(INC) $(RT) $(APP) app/plat_host.c -o $@
$(MODEL)/nslm_test.h: $(MODEL)/nslm_app $(MODEL)/model.nslm
	echo "#define NSLM_TEST_HASH 0x$$(NSLM_DIR=$(MODEL) $(MODEL)/nslm_app --hash)u" > $@

.PHONY: host arm app-host
