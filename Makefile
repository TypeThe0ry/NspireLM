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

ASM = runtime/arm/matmul_q8.S runtime/arm/matmul_q4.S

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

# Headless app (scripted keys, frames to PPM). The expected self-test hash is produced by a build
# without it (nslm_hashgen) and baked into every app build, so the self-test proves bit-exactness.
APPSRC = $(RT) $(APP) app/plat_host.c
APPDEP = $(APPSRC) app/plat.h runtime/nslm.h $(MODEL)/nslm_model.h
app-host: $(MODEL)/nslm_app $(MODEL)/nslm_app_arm
$(MODEL)/nslm_hashgen: $(APPDEP)
	$(CC) $(CFLAGS) $(WARN) -D_POSIX_C_SOURCE=200112L -DNSLM_NO_TEST_H $(INC) $(APPSRC) -o $@
$(MODEL)/nslm_test.h: $(MODEL)/nslm_hashgen $(MODEL)/model.nslm
	echo "#define NSLM_TEST_HASH 0x$$(NSLM_DIR=$(MODEL) $(MODEL)/nslm_hashgen --hash)u" > $@
$(MODEL)/nslm_app: $(APPDEP) $(MODEL)/nslm_test.h
	$(CC) $(CFLAGS) $(WARN) -D_POSIX_C_SOURCE=200112L $(INC) $(APPSRC) -o $@
$(MODEL)/nslm_app_arm: $(APPDEP) $(ASM) $(MODEL)/nslm_test.h
	$(ARMCC) $(CFLAGS) $(WARN) -D_POSIX_C_SOURCE=200112L -DNSLM_HAVE_ASM_MATMUL -mcpu=arm926ej-s -marm -static \
	  $(INC) $(APPSRC) $(ASM) -o $@

# TI-Nspire CX II program (Ndless SDK). Output directory $(MODEL)/nspire holds everything to copy:
# nslm.tns (program), model.nslm.tns, tok.nslt.tns.
NDLESS ?= $(HOME)/ndless/ndless-sdk
NSPIRE_ENV = PATH="$(NDLESS)/bin:$(NDLESS)/toolchain/install/bin:$$PATH"
NSPIRE_FLAGS = -O2 -marm -std=gnu11 -Wall -Wextra -Wno-unused-function -DNSLM_HAVE_ASM_MATMUL $(INC) -Iapp
NSPIRE_SRC = $(RT) $(APP) app/plat_nspire.c
nspire: $(MODEL)/nspire/nslm.tns
$(MODEL)/nspire/nslm.tns: $(NSPIRE_SRC) $(ASM) app/plat.h runtime/nslm.h $(MODEL)/nslm_model.h $(MODEL)/nslm_test.h
	mkdir -p $(MODEL)/nspire
	cd $(MODEL)/nspire && for f in $(abspath $(NSPIRE_SRC) $(ASM)); do \
	  $(NSPIRE_ENV) nspire-gcc $(NSPIRE_FLAGS:-I%=-I$(CURDIR)/%) -c $$f -o $$(basename $$f).o || exit 1; done
	cd $(MODEL)/nspire && $(NSPIRE_ENV) nspire-ld *.o -o nslm.elf
	$(NSPIRE_ENV) genzehn --input $(MODEL)/nspire/nslm.elf --output $(MODEL)/nspire/nslm.zehn \
	  --name "NspireLM" --author "NspireLM" --notice "offline language model" \
	  --240x320-support true --uses-lcd-blit true
	$(NSPIRE_ENV) make-prg $(MODEL)/nspire/nslm.zehn $@
	rm -f $(MODEL)/nspire/nslm.zehn
	cp $(MODEL)/model.nslm $(MODEL)/nspire/model.nslm.tns
	cp $(MODEL)/tok.nslt $(MODEL)/nspire/tok.nslt.tns

.PHONY: host arm app-host nspire
