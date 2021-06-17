#include "openssl/md5.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* calc and print MD5 checksum of a memory chunk */
char* getMD5DigestStr(void* buf, size_t buflen)
{
  MD5_CTX ctx;
  unsigned char sum[MD5_DIGEST_LENGTH];
  int i;
  char* retval;
  char* ptr;

  MD5_Init(&ctx);
  MD5_Update(&ctx, buf, buflen);
  MD5_Final(sum, &ctx);

  retval = calloc(MD5_DIGEST_LENGTH * 2 + 1, sizeof(*retval));
  ptr = retval;

  for (i = 0; i < MD5_DIGEST_LENGTH; i++) {
    snprintf(ptr, 3, "%02X", sum[i]);
    ptr += 2;
  }

  return retval;
}


