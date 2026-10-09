/*!
\file  
\brief Various routines for providing portable 32 and 64 bit random number
       generators.

\date   Started 5/17/2007
\author George
\version\verbatim $Id: random.c 11793 2012-04-04 21:03:02Z karypis $ \endverbatim
*/

#include <GKlib.h>

/* Modified for pantheonsim (see ../README.md): the non-USE_GKRAND path draws
   from libc's rand(), a process-wide generator that gk_randinit() reseeds. In
   a shim loaded into someone else's process that would reset the host
   program's own rand() sequence on every call, and two threads would share
   one state. glibc's rand() is random() on a TYPE_3 state seeded by srand();
   random_r() on a thread-local state of the same type, seeded by
   initstate_r(), draws the identical sequence, so the permutations METIS
   returns are the ones an unmodified METIS (the one NVIDIA's cuSOLVER links)
   returns, and nothing else sees the generator. */
#include <stdlib.h>
#include <string.h>
static __thread struct random_data vgpu_rd;
static __thread char vgpu_rstate[128];
static __thread int vgpu_rseeded = 0;
static int vgpu_rand(void)
{
  int32_t r = 0;
  if (!vgpu_rseeded) {   /* srand()'s default seed is 1 */
    memset(&vgpu_rd, 0, sizeof vgpu_rd);
    initstate_r(1u, vgpu_rstate, sizeof vgpu_rstate, &vgpu_rd);
    vgpu_rseeded = 1;
  }
  random_r(&vgpu_rd, &r);
  return (int)r;
}
static void vgpu_srand(unsigned int seed)
{
  memset(&vgpu_rd, 0, sizeof vgpu_rd);
  initstate_r(seed, vgpu_rstate, sizeof vgpu_rstate, &vgpu_rd);
  vgpu_rseeded = 1;
}


/*************************************************************************/
/*! Create the various random number functions */
/*************************************************************************/
GK_MKRANDOM(gk_c,   size_t, char)
GK_MKRANDOM(gk_i,   size_t, int)
GK_MKRANDOM(gk_f,   size_t, float)
GK_MKRANDOM(gk_d,   size_t, double)
GK_MKRANDOM(gk_idx, size_t, gk_idx_t)
GK_MKRANDOM(gk_z,   size_t, ssize_t)



/*************************************************************************/
/*! GKlib's built in random number generator for portability across 
    different architectures */
/*************************************************************************/
#ifdef USE_GKRAND
/* 
   A C-program for MT19937-64 (2004/9/29 version).
   Coded by Takuji Nishimura and Makoto Matsumoto.

   This is a 64-bit version of Mersenne Twister pseudorandom number
   generator.

   Before using, initialize the state by using init_genrand64(seed)  
   or init_by_array64(init_key, key_length).

   Copyright (C) 2004, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define NN 312
#define MM 156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define LM 0x7FFFFFFFULL /* Least significant 31 bits */


/* The array for the state vector */
static uint64_t mt[NN]; 
/* mti==NN+1 means mt[NN] is not initialized */
static int mti=NN+1; 
#endif /* USE_GKRAND */

/* initializes mt[NN] with a seed */
void gk_randinit(uint64_t seed)
{
#ifdef USE_GKRAND
  mt[0] = seed;
  for (mti=1; mti<NN; mti++) 
    mt[mti] = (6364136223846793005ULL * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti);
#else
  vgpu_srand((unsigned int) seed);
#endif
}


/* generates a random number on [0, 2^64-1]-interval */
uint64_t gk_randint64(void)
{
#ifdef USE_GKRAND
  int i;
  unsigned long long x;
  static uint64_t mag01[2]={0ULL, MATRIX_A};

  if (mti >= NN) { /* generate NN words at one time */
    /* if init_genrand64() has not been called, */
    /* a default initial seed is used     */
    if (mti == NN+1) 
      gk_randinit(5489ULL); 

    for (i=0; i<NN-MM; i++) {
      x = (mt[i]&UM)|(mt[i+1]&LM);
      mt[i] = mt[i+MM] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
    }
    for (; i<NN-1; i++) {
      x = (mt[i]&UM)|(mt[i+1]&LM);
      mt[i] = mt[i+(MM-NN)] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
    }
    x = (mt[NN-1]&UM)|(mt[0]&LM);
    mt[NN-1] = mt[MM-1] ^ (x>>1) ^ mag01[(int)(x&1ULL)];

    mti = 0;
  }

  x = mt[mti++];

  x ^= (x >> 29) & 0x5555555555555555ULL;
  x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
  x ^= (x << 37) & 0xFFF7EEE000000000ULL;
  x ^= (x >> 43);

  return x & 0x7FFFFFFFFFFFFFFF;
#else
  /* The two draws of the original, `(rand() << 32) | rand()`, are unsequenced
     in C; the order is fixed here (high word first). */
  uint64_t hi = (uint64_t) vgpu_rand();
  uint64_t lo = (uint64_t) vgpu_rand();
  return (uint64_t)(hi << 32 | lo);
#endif
}

/* generates a random number on [0, 2^32-1]-interval */
uint32_t gk_randint32(void)
{
#ifdef USE_GKRAND
  return (uint32_t)(gk_randint64() & 0x7FFFFFFF);
#else
  return (uint32_t)vgpu_rand();
#endif
}


