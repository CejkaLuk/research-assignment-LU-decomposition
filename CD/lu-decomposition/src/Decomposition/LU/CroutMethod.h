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

namespace Decomposition {
namespace LU {

class CroutMethod
{
    public:

        CroutMethod();

        template< typename Matrix1, typename Matrix2, typename Matrix3 >
        static void decompose( Matrix1& A, Matrix2& L, Matrix3& U );

        template< typename Matrix1, typename Matrix2 >
        static void decompose( Matrix1& A, Matrix2& Z );

        static void print();

};

} // namespace LU
} // namespace Decomposition

#include <Decomposition/LU/CroutMethod.hpp>