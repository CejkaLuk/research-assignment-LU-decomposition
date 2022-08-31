/***************************************************************************
                          CroutMethodTest.h  -  description
                             -------------------
    begin                : Jul 18, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <Decomposition/LU/CroutMethod.h>

namespace Decomposition {
namespace LU {

CroutMethod::CroutMethod()
{
}

template< typename Matrix1, typename Matrix2, typename Matrix3 >
void CroutMethod::decompose( Matrix1& A, Matrix2& L, Matrix3& U )
{
    using RealType  = typename Matrix1::RealType;
    using IndexType = typename Matrix1::IndexType;

    IndexType num_rows = A.getRows();
    IndexType num_cols = A.getColumns();

    TNL_ASSERT_EQ( num_rows, num_cols, "Matrix A must be a square matrix!" );

    IndexType i, j, k;
    RealType sum = 0;

    for( j = 0; j < num_rows; ++j )
    {
        for( i = j; i < num_rows; ++i )
        {
            sum = 0;
            for( k = 0; k < j; ++k )
                sum = sum + L.getElement( i, k ) * U.getElement( k, j );

            L.setElement( i, j, A.getElement( i, j ) - sum );
        }

        for( i = j; i < num_rows; ++i )
        {
            TNL_ASSERT( L.getElement( j, j ) != 0,
                        std::cerr << "L( " << i << ", " << j <<
                        " ) = 0. Cannot divide by 0." << std::endl );
            sum = 0;
            for( k = 0; k < j; ++k )
                sum = sum + L.getElement( j, k ) * U.getElement( k, i );

            U.setElement( j, i, ( A.getElement( j, i ) - sum ) / L.getElement( j, j ) );
        }
    }

}

// WARNING: 1s found on the main diagonal of U (Crout decomposition) are not stored in matrix Z.
// Matrix Z has elements of L on its main diagonal (where L comes from LU decomposition of matrix A).
template< typename Matrix1, typename Matrix2 >
void CroutMethod::decompose( Matrix1& A, Matrix2& Z )
{
    using RealType  = typename Matrix1::RealType;
    using IndexType = typename Matrix1::IndexType;

    IndexType num_rows = A.getRows();
    IndexType num_cols = A.getColumns();

    TNL_ASSERT_EQ( num_rows, num_cols, "Matrix A must be a square matrix!" );

    IndexType i, j, k;
    RealType sum = 0;

    for( j = 0; j < num_rows; ++j )
    {
        for( i = j; i < num_rows; ++i )
        {
            sum = 0;
            for( k = 0; k < j; ++k )
                sum = sum + Z.getElement( i, k ) * Z.getElement( k, j );

            Z.setElement( i, j, A.getElement( i, j ) - sum );
        }

        for( i = j; i < num_rows; ++i )
        {
            if( j == i ) continue;

            TNL_ASSERT( Z.getElement( j, j ) != 0,
                        std::cerr << "Z( " << j << ", " << j <<
                        " ) = 0. Cannot divide by 0." << std::endl );
            sum = 0;
            for( k = 0; k < j; ++k )
                sum = sum + Z.getElement( j, k ) * Z.getElement( k, i );

            Z.setElement( j, i, ( A.getElement( j, i ) - sum ) / Z.getElement( j, j ) );
        }
    }
}

void CroutMethod::print()
{
    std::cout << "Printing LU Crout method" << std::endl;
}

} // namespace LU
} // namespace Decomposition
