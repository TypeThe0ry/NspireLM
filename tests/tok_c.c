/* Encodes each stdin line (with "\n" escapes and "\\" decoded) with a tok.nslt; prints model ids. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nslm.h"

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *img = malloc((size_t)len);
    if (fread(img, 1, (size_t)len, f) != (size_t)len) return 1;
    fclose(f);
    nslm_tok t;
    if (nslm_tok_bind(&t, img, (size_t)len)) {
        fprintf(stderr, "bad tokenizer\n");
        return 1;
    }
    static char line[1 << 16], text[1 << 16], scratch[256];
    static int ids[(1 << 16) + 3];
    while (fgets(line, sizeof line, stdin)) {
        size_t k = 0;
        for (size_t i = 0; line[i] && line[i] != '\n'; i++) {
            if (line[i] == '\\' && line[i + 1] == 'n') text[k++] = '\n', i++;
            else if (line[i] == '\\' && line[i + 1] == 't') text[k++] = '\t', i++;
            else if (line[i] == '\\' && line[i + 1] == '\\') text[k++] = '\\', i++;
            else text[k++] = line[i];
        }
        text[k] = 0;
        int n = nslm_tok_encode(&t, text, ids, scratch);
        for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", ids[i]);
        printf("\n");
    }
    free(img);
    return 0;
}
