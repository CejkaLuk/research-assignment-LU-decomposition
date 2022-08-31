/***************************************************************************
                          CroutMethodTest.h  -  description
                             -------------------
    begin                : Jul 18, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/DenseMatrix.h>

#define BLOCK_SIZE 8

namespace Decomposition {
namespace LU {

class CroutMethodIterative
{
    public:

        CroutMethodIterative();

        template< typename Matrix1,
                  typename Matrix2,
                  typename Matrix3 >
        static void decompose( Matrix1& A, Matrix2& L, Matrix3& U, const int threads );

        template< typename Matrix1,
                  typename Matrix2 >
        static void decompose( Matrix1& A, Matrix2& Z, const int threads );

        static void print();
};

} // namespace LU
} // namespace Decomposition

#include <Decomposition/LU/CroutMethodIterative.hpp>