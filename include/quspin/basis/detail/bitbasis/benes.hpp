// Copyright 2024 Phillip Weinberg
#pragma once

#include <quspin/basis/detail/bitbasis/info.hpp>

namespace quspin::details::basis::benes {

static const int no_index = -1;

template<typename I>
struct tr_bfly {
    enum { ld_bits = bit_info<I>::ld_bits };

    // This structure is used to hold the configuration of
    // butterfly-based operations as well as compress and expand.
    typename bit_info<I>::value_type cfg[ld_bits];  // butterfly configuration
    typename bit_info<I>::value_type mask;  // saved mask, for compress/expand

    // Here is sketched how to convert this to a class:
    // Include all the generator and usage functions as private methods
    // and replace the parameter self by the implicit object pointer this.
    // Add the many compound routines.
    // Remove the name suffix  for all methods.
    // If you want to cache the configuration, add here:
    //   kind: the generator kind
    //     enum (initialized, frot, vrot, ce_right, ce_left, cef_right,
    //     cef_left)
    //   sw: the used subword size (t_subword)
    // Add an initializer/constructor which sets kind to initialized.
    // The generator routines must set the keys (kind, mask, sw).
    // The compound routines check the cached keys (kind, mask, sw);
    //   if not equal, call the generator routine and update the configuration;
    // finally they call the usage routine.
};

template<typename I>
struct tr_benes {
    tr_bfly<I> b1, b2;
};

template<typename I>  // primary template
struct ta_index {
    enum { bits = bit_info<I>::bits };

    int data[bits];

    int& operator[](int idx) { return data[idx]; }

    int operator[](int idx) const { return data[idx]; }
};

template<typename I>  // primary template
struct ta_subword {
    enum { ld_bits = bit_info<I>::ld_bits };

    unsigned int data[ld_bits];

    unsigned int& operator[](int idx) { return data[idx]; }

    unsigned int operator[](int idx) const { return data[idx]; }
};

//////
// aux functions

template<typename I>
void invert_perm(const ta_index<I>& src, ta_index<I>& tgt) {
  int i;

  for (i = 0; i <= ta_index<I>::bits - 1; ++i) {
    tgt[i] = no_index;
  }
  for (i = 0; i <= ta_index<I>::bits - 1; ++i) {
    if (src[i] != no_index) {
      tgt[src[i]] = i;
    }
  }
}

template<typename I>
I bit_permute_step(I x, const typename bit_info<I>::value_type& m,
                   const int shift) {
  // INLINE
  // Can be replaced by bit_permute_step_simple,
  // if for the relevant bits n the following holds:
  // nr_1bits(bit_permute_step_simple(n,m,shift)) == nr_1bits(n)
  // x86: >= 6/5 cycles
  // ARM: >= 4/4 cycles

  // assert(((m << shift) & m) == 0);
  // assert(((m << shift) >> shift) == m);
  I t = ((x >> shift) ^ x) & m;
  x = x ^ t;
  t = t << shift;
  x = x ^ t;  // x = (x ^ t) ^ (t << shift);
  return x;
}

//////
// Butterfly network
template<typename I>
I bfly(const tr_bfly<I>* self, I x) {
  // Apply butterfly network on x configured by
  //   - gen_frot
  //   - gen_vrot
  //   - gen_cef_right
  //   - gen_cef_left

  int stage, j;

  for (stage = bit_info<I>::ld_bits - 1; stage >= 0; --stage) {  // UNROLL
    // x = butterfly(x, self->cfg[stage], sw);
    j = 1 << stage;
    x = bit_permute_step<I>(x, self->cfg[stage], j);
  }

  return x;
}

template<typename I>
I ibfly(const tr_bfly<I>* self, I x) {
  // Apply inverse butterfly network on x configured by
  //   - gen_frot
  //   - gen_vrot
  //   - gen_cef_right
  //   - gen_cef_left

  int stage, j;

  for (stage = 0; stage <= bit_info<I>::ld_bits - 1; ++stage) {  // UNROLL
    // x = butterfly(x, self->cfg[stage], sw);
    j = 1 << stage;
    x = bit_permute_step<I>(x, self->cfg[stage], j);
  }

  return x;
}

//////
// Permutations via benes_perm network

template<typename t_bit_index>
void exchange_bit_index(t_bit_index* a, t_bit_index* b) {
  // INLINE

  t_bit_index q;

  q = *a;
  *a = *b;
  *b = q;
}

template<typename I>
void gen_benes_ex(tr_benes<I>* self, const ta_index<I>& c_tgt,
                  const ta_subword<I>& a_stage) {
  // Generate a configuration for the benes_perm network with variable stage
  // order. Use benes_fwd_ex and benes_bwd_ex. Algorithm as sketched by Donal
  // E. Knuth,
  //   The art of computer programming, vol. 4, pre-fascicle 1a
  // Implemented 2011-12-02 by Jasper L. Neumann
  // Modified 2012-08-31 to allow for "don't care" entries.
  // Modified 2018-05-22 to allow for templated types only for
  // benes'/butterfly networks.
  // Modified 2018-05-24 to allow for templated types only for
  // benes'/butterfly networks.

  ta_index<I> src, inv_src;
  ta_index<I> tgt, inv_tgt;
  typename bit_info<I>::value_type cfg_src, cfg_tgt;
  typename bit_info<I>::value_type src_set;
  typename bit_info<I>::value_type lo_bit = 1;

  int stage;
  int mask;
  int main_idx, aux_idx, src_idx, tgt_idx, idx2;
  int stage_idx;
  int s;

  for (s = 0; s <= bit_info<I>::bits - 1; ++s) {
    src[s] = no_index;
    tgt[s] = no_index;
  }
  for (s = 0; s <= bit_info<I>::bits - 1; ++s) {
    if (c_tgt[s] != no_index) {
      tgt[s] = s;
      src[c_tgt[s]] = s;
    }
  }
  invert_perm(src, inv_src);
  invert_perm(tgt, inv_tgt);
  for (stage_idx = 0; stage_idx <= bit_info<I>::ld_bits - 1; ++stage_idx) {
    stage = a_stage[stage_idx];
    src_set = 0;
    src_idx = 0;
    mask = ((int)lo_bit) << stage;
    cfg_src = 0;
    cfg_tgt = 0;
    for (main_idx = 0; main_idx <= bit_info<I>::bits - 1;
         ++main_idx) {               // This order to meet Waksman test
      if ((main_idx & mask) == 0) {  // low only
        for (aux_idx = 0; aux_idx <= 1; ++aux_idx) {
          src_idx = main_idx + (aux_idx << stage);
          if (((lo_bit << src_idx) & src_set) == 0) {  // yet unhandled
            if (src[src_idx] != no_index) {            // not open

              do {
                src_set = (src_set | (lo_bit << src_idx));
                tgt_idx = inv_tgt[src[src_idx]];
                if (tgt[tgt_idx] == no_index) {
                  break;  // open end
                }

                if (((src_idx ^ tgt_idx) & mask) == 0) {
                  // straight
                  tgt_idx = tgt_idx ^ mask;
                } else {
                  // cross
                  cfg_tgt = (cfg_tgt | (lo_bit << (tgt_idx & ~mask)));
                  idx2 = (tgt_idx ^ mask);
                  exchange_bit_index(&tgt[tgt_idx], &tgt[idx2]);
                  inv_tgt[tgt[idx2]] = idx2;
                  if (tgt[tgt_idx] != no_index) {
                    inv_tgt[tgt[tgt_idx]] = tgt_idx;
                  }
                }

                if (tgt[tgt_idx] == no_index) {
                  break;  // open end
                }

                src_idx = inv_src[tgt[tgt_idx]];

                if (((src_idx ^ tgt_idx) & mask) == 0) {
                  // straight
                  src_set = (src_set | (lo_bit << src_idx));
                  src_idx = (src_idx ^ mask);
                } else {
                  // cross

                  cfg_src = (cfg_src | (lo_bit << (src_idx & ~mask)));
                  idx2 = (src_idx ^ mask);
                  src_set = (src_set | (lo_bit << idx2));

                  exchange_bit_index(&src[src_idx], &src[idx2]);
                  inv_src[src[idx2]] = idx2;
                  if (src[src_idx] != no_index) {
                    inv_src[src[src_idx]] = src_idx;
                  }
                }
                if (src[src_idx] == no_index) {
                  break;  // open end
                }
                if (((lo_bit << src_idx) & src_set) != 0) {  // yet unhandled
                  break;                                     // already handled
                }
              } while (true);
            }
          }
        }
      }
    }
    self->b1.cfg[stage] = cfg_src;
    self->b2.cfg[stage] = cfg_tgt;
  }
  // Reduce inner stages to one (not needed)
  // self->b2.cfg[0] = self->b2.cfg[0] ^ self->b1.cfg[0];
  // self->b1.cfg[0] = 0;
}

template<typename I>
void gen_benes(tr_benes<I>* self, const ta_index<I>& c_tgt) {
  // INLINE
  // Generate a configuration for the standard benes_perm network.
  ta_subword<I> a_stage_bwd;
  for (unsigned int i = 0; i < bit_info<I>::ld_bits; ++i) {
    a_stage_bwd[i] = bit_info<I>::ld_bits - i - 1;
  }
  gen_benes_ex(self, c_tgt, a_stage_bwd);  // standard benes_perm order
}

template<typename I>
I benes_fwd(const tr_benes<I>* self, I x) {
  // Apply benes_perm network.
  // c_tgt of gen_benes selected source indexes.

  return ibfly(&self->b2, bfly(&self->b1, x));
}

template<typename I>
I benes_bwd(const tr_benes<I>* self, I x) {
  // Apply benes_perm network.
  // c_tgt of gen_benes selected target indexes.

  return ibfly(&self->b1, bfly(&self->b2, x));
}
}  // namespace quspin::details::basis::benes

#ifdef QUSPIN_UNIT_TESTS

namespace quspin::details::basis::benes {  // explicit instantiation for code
                                           // coverage

template struct tr_bfly<uint8_t>;
template struct tr_benes<uint8_t>;
template struct ta_index<uint8_t>;
template struct ta_subword<uint8_t>;

template void invert_perm<uint8_t>(const ta_index<uint8_t>&,
                                   ta_index<uint8_t>&);
template uint8_t bit_permute_step<uint8_t>(uint8_t, uint8_t, int);
template uint8_t bfly<uint8_t>(const tr_bfly<uint8_t>*, uint8_t);
template uint8_t ibfly<uint8_t>(const tr_bfly<uint8_t>*, uint8_t);
template void exchange_bit_index<uint8_t>(uint8_t* a, uint8_t* b);
template void gen_benes_ex<uint8_t>(tr_benes<uint8_t>*,
                                    const ta_index<uint8_t>&,
                                    const ta_subword<uint8_t>&);
template void gen_benes<uint8_t>(tr_benes<uint8_t>*, const ta_index<uint8_t>&);
template uint8_t benes_fwd<uint8_t>(const tr_benes<uint8_t>*, uint8_t);
template uint8_t benes_bwd<uint8_t>(const tr_benes<uint8_t>*, uint8_t);

}  // namespace quspin::details::basis::benes

TEST_SUITE("quspin/basis/bitbasis/benes.h") {
  using namespace quspin::details::basis::benes;

  TEST_CASE("ta_subword set/get") {
    ta_subword<uint8_t> subword;

    for (int i = 0; i < quspin::details::basis::bit_info<uint8_t>::ld_bits;
         ++i) {
      subword[i] = i;
      CHECK(subword[i] == i);
    }
  }

  TEST_CASE("ta_index set/get") {
    ta_index<uint8_t> index;

    for (int i = 0; i < quspin::details::basis::bit_info<uint8_t>::bits; ++i) {
      index[i] = i;
      CHECK(index[i] == i);
    }
  }

  TEST_CASE("invert_perm") {
    ta_index<uint8_t> i1;
    ta_index<uint8_t> i2;

    for (int i = 0; i < quspin::details::basis::bit_info<uint8_t>::bits; ++i) {
      i1[i] = no_index;
    }

    i1[0] = 1;
    i1[1] = 3;
    i1[2] = 0;
    i1[3] = 2;

    invert_perm<uint8_t>(i1, i2);

    CHECK(i2[0] == 2);
    CHECK(i2[1] == 0);
    CHECK(i2[2] == 3);
    CHECK(i2[3] == 1);
    CHECK(i2[4] == no_index);
    CHECK(i2[5] == no_index);
    CHECK(i2[6] == no_index);
    CHECK(i2[7] == no_index);
  }
}

#endif
