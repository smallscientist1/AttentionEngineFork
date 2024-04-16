/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tl/op/bulk_copy.h
 * \brief Bulk copy operator.
 *
 */

#ifndef TVM_TL_OP_BULK_COPY_H_
#define TVM_TL_OP_BULK_COPY_H_

#include "elem.h"

namespace tvm {
namespace tl {

using namespace tir;

struct TMADesc {
  size_t rank;
  int data_type;
  Array<PrimExpr> global_shape, global_stride;
  Array<PrimExpr> smem_box, smem_stride;
  PrimExpr global_addr;
  int swizzle;
  int interleave;
  int oob_fill;
  int l2_promotion;

  Array<PrimExpr> EncodeCallArgs() const;
};

DataType cuTensorMapType();

/*!
 * \brief tvm intrinsics for TMADescriptor creation
 *
 * CreateTMADescriptorOp(data_type, rank, global_addr, global_shape..., global_stride...,
 * smem_box..., smem_stride..., interleave, swizzle, l2_promotion, oob_fill)
 *
 */
const Op& CreateTMADescriptorOp();

/*!
 * \brief tvm intrinsics for bulk async copy using mbarrier and TMADescriptor
 *
 * TMACopyOp(descritor, barrier_id, smem_data, is_load, coord_0, coord_1, ...)
 *
 */
const Op& TMACopyOp();

/*!
 * \brief tvm intrinsics for mbarrier wait with parity bit
 *
 * MBarrierWaitParity(barrier_id, parity)
 *
 */
const Op& MBarrierWaitParity();

/*!
 * \brief tvm intrinsics for ldmatrix
 *
 * LDMatrixOp(transposed, num, shared_addr, local_addr)
 *
 */
const Op& LDMatrixOp();

}  // namespace tl
}  // namespace tvm

#endif  //  TVM_TL_OP_BULK_COPY_H_