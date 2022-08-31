/***************************************************************************
                          LUIterativeCroutMethodTest.h  -  description
                             -------------------
    begin                : Oct 24, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <Decomposition/LU/CroutMethodIterative.h>

#include "UnitTests/utils/SampleMatrices.h"

#ifdef HAVE_GTEST
#include "gtest/gtest.h"

template< typename Matrix1, typename Matrix2 >
void verifyDecompositionResult( Matrix1& L, Matrix2& U,
                                std::vector< typename Matrix1::RealType >& lRowMajor,
                                std::vector< typename Matrix1::RealType >& uRowMajor,
                                typename Matrix1::RealType percentTolerance = 0.001 )
{
    using RealType  = typename Matrix1::RealType;
    using IndexType = typename Matrix1::IndexType;

    for( IndexType i = 0; i < L.getRows(); ++i )
    {
        for( IndexType j = 0; j < L.getColumns(); ++j )
        {
            // Default tolerance from value: 0.1% (0.001)
            RealType tolerance = std::abs( lRowMajor[ i*L.getRows() + j ] * percentTolerance);
            EXPECT_NEAR( L.getElement( i, j ), lRowMajor[ i*L.getRows() + j ], tolerance ) <<
                "Expected L( " << i << ", " << j << " ) = " << lRowMajor[ i*L.getRows() + j ] << "\n" <<
                "Actual value: " << L.getElement( i, j );

            tolerance = std::abs( uRowMajor[ i*U.getRows() + j ] * percentTolerance);
            EXPECT_NEAR( U.getElement( i, j ), uRowMajor[ i*U.getRows() + j ], tolerance ) <<
                "Expected U( " << i << ", " << j << " ) = " << uRowMajor[ i*U.getRows() + j ] << "\n" <<
                "Actual value: " << U.getElement( i, j );
        }
    }
};

template< typename Matrix >
void verifyDecompositionResult( Matrix& Z,
                                std::vector< typename Matrix::RealType >& zRowMajor,
                                typename Matrix::RealType percentTolerance = 0.001 )
{
    using RealType  = typename Matrix::RealType;
    using IndexType = typename Matrix::IndexType;

    for( IndexType i = 0; i < Z.getRows(); ++i )
    {
        for( IndexType j = 0; j < Z.getColumns(); ++j )
        {
            RealType tolerance = std::abs( zRowMajor[ i*Z.getRows() + j ] * percentTolerance);
            EXPECT_NEAR( Z.getElement( i, j ), zRowMajor[ i*Z.getRows() + j ], tolerance ) <<
                "Expected Z( " << i << ", " << j << " ) = " << zRowMajor[ i*Z.getRows() + j ] << "\n" <<
                "Actual value: " << Z.getElement( i, j );
        }
    }
};

template< typename Matrix >
void test_CroutMethodSanity()
{
    using IndexType = typename Matrix::IndexType;

    const IndexType rows = 9;
    const IndexType cols = 8;

    Matrix m;
    m.setDimensions( rows, cols );

    EXPECT_EQ( m.getRows(), rows );
    EXPECT_EQ( m.getColumns(), cols );

    Decomposition::LU::CroutMethodIterative::print();
}

template< typename Matrix >
void test_CroutMethod2x2MatrixDecompose()
{
    const Matrix A( {
        { 4, 24 },
        { 2, 15 }
    } );

    Matrix L, U;
    L.setLike( A );
    U.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, L, U, BLOCK_SIZE );

    EXPECT_EQ( L.getElement( 0, 0 ),  4.0 );
    EXPECT_EQ( L.getElement( 0, 1 ),  0.0 );
    EXPECT_EQ( L.getElement( 1, 0 ),  2.0 );
    EXPECT_EQ( L.getElement( 1, 1 ),  3.0 );

    EXPECT_EQ( U.getElement( 0, 0 ),  1.0 );
    EXPECT_EQ( U.getElement( 0, 1 ),  6.0 );
    EXPECT_EQ( U.getElement( 1, 0 ),  0.0 );
    EXPECT_EQ( U.getElement( 1, 1 ),  1.0 );

    Matrix Z;
    Z.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, Z, BLOCK_SIZE );

    EXPECT_EQ( Z.getElement( 0, 0 ),  4.0 );
    EXPECT_EQ( Z.getElement( 0, 1 ),  6.0 );
    EXPECT_EQ( Z.getElement( 1, 0 ),  2.0 );
    EXPECT_EQ( Z.getElement( 1, 1 ),  3.0 );
}

template< typename Matrix >
void test_CroutMethod3x3MatrixDecompose()
{
    const Matrix A( {
        { 2,  4,  6 },
        { 1,  6, 19 },
        { 5, 13, 33 }
    } );

    Matrix L, U;
    L.setLike( A );
    U.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, L, U, BLOCK_SIZE );

    EXPECT_EQ( L.getElement( 0, 0 ),  2.0 );
    EXPECT_EQ( L.getElement( 0, 1 ),  0.0 );
    EXPECT_EQ( L.getElement( 0, 2 ),  0.0 );
    EXPECT_EQ( L.getElement( 1, 0 ),  1.0 );
    EXPECT_EQ( L.getElement( 1, 1 ),  4.0 );
    EXPECT_EQ( L.getElement( 1, 2 ),  0.0 );
    EXPECT_EQ( L.getElement( 2, 0 ),  5.0 );
    EXPECT_EQ( L.getElement( 2, 1 ),  3.0 );
    EXPECT_EQ( L.getElement( 2, 2 ),  6.0 );

    EXPECT_EQ( U.getElement( 0, 0 ),  1.0 );
    EXPECT_EQ( U.getElement( 0, 1 ),  2.0 );
    EXPECT_EQ( U.getElement( 0, 2 ),  3.0 );
    EXPECT_EQ( U.getElement( 1, 0 ),  0.0 );
    EXPECT_EQ( U.getElement( 1, 1 ),  1.0 );
    EXPECT_EQ( U.getElement( 1, 2 ),  4.0 );
    EXPECT_EQ( U.getElement( 2, 0 ),  0.0 );
    EXPECT_EQ( U.getElement( 2, 1 ),  0.0 );
    EXPECT_EQ( U.getElement( 2, 2 ),  1.0 );

    Matrix Z;
    Z.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, Z, BLOCK_SIZE );

    EXPECT_EQ( Z.getElement( 0, 0 ),  2.0 );
    EXPECT_EQ( Z.getElement( 0, 1 ),  2.0 );
    EXPECT_EQ( Z.getElement( 0, 2 ),  3.0 );
    EXPECT_EQ( Z.getElement( 1, 0 ),  1.0 );
    EXPECT_EQ( Z.getElement( 1, 1 ),  4.0 );
    EXPECT_EQ( Z.getElement( 1, 2 ),  4.0 );
    EXPECT_EQ( Z.getElement( 2, 0 ),  5.0 );
    EXPECT_EQ( Z.getElement( 2, 1 ),  3.0 );
    EXPECT_EQ( Z.getElement( 2, 2 ),  6.0 );
}

template< typename Matrix >
void test_CroutMethod4x4SemiSparseMatrixDecompose()
{
    const Matrix A( {
        { 2,  6,  0,  0 },
        { 3, -3, 12,  0 },
        { 1,  4,  3,  0 },
        { 0, 10, 24, 30 }
    } );

    Matrix L, U;
    L.setLike( A );
    U.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, L, U, BLOCK_SIZE );

    EXPECT_EQ( L.getElement( 0, 0 ),   2.0 );
    EXPECT_EQ( L.getElement( 0, 1 ),   0.0 );
    EXPECT_EQ( L.getElement( 0, 2 ),   0.0 );
    EXPECT_EQ( L.getElement( 0, 3 ),   0.0 );
    EXPECT_EQ( L.getElement( 1, 0 ),   3.0 );
    EXPECT_EQ( L.getElement( 1, 1 ), -12.0 );
    EXPECT_EQ( L.getElement( 1, 2 ),   0.0 );
    EXPECT_EQ( L.getElement( 1, 3 ),   0.0 );
    EXPECT_EQ( L.getElement( 2, 0 ),   1.0 );
    EXPECT_EQ( L.getElement( 2, 1 ),   1.0 );
    EXPECT_EQ( L.getElement( 2, 2 ),   4.0 );
    EXPECT_EQ( L.getElement( 2, 3 ),   0.0 );
    EXPECT_EQ( L.getElement( 3, 0 ),   0.0 );
    EXPECT_EQ( L.getElement( 3, 1 ),  10.0 );
    EXPECT_EQ( L.getElement( 3, 2 ),  34.0 );
    EXPECT_EQ( L.getElement( 3, 3 ),  30.0 );

    EXPECT_EQ( U.getElement( 0, 0 ),   1.0 );
    EXPECT_EQ( U.getElement( 0, 1 ),   3.0 );
    EXPECT_EQ( U.getElement( 0, 2 ),   0.0 );
    EXPECT_EQ( U.getElement( 0, 3 ),   0.0 );
    EXPECT_EQ( U.getElement( 1, 0 ),   0.0 );
    EXPECT_EQ( U.getElement( 1, 1 ),   1.0 );
    EXPECT_EQ( U.getElement( 1, 2 ),  -1.0 );
    EXPECT_EQ( U.getElement( 1, 3 ),   0.0 );
    EXPECT_EQ( U.getElement( 2, 0 ),   0.0 );
    EXPECT_EQ( U.getElement( 2, 1 ),   0.0 );
    EXPECT_EQ( U.getElement( 2, 2 ),   1.0 );
    EXPECT_EQ( U.getElement( 2, 3 ),   0.0 );
    EXPECT_EQ( U.getElement( 3, 0 ),   0.0 );
    EXPECT_EQ( U.getElement( 3, 1 ),   0.0 );
    EXPECT_EQ( U.getElement( 3, 2 ),   0.0 );
    EXPECT_EQ( U.getElement( 3, 3 ),   1.0 );

    Matrix Z;
    Z.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, Z, BLOCK_SIZE );

    EXPECT_EQ( Z.getElement( 0, 0 ),   2.0 );
    EXPECT_EQ( Z.getElement( 0, 1 ),   3.0 );
    EXPECT_EQ( Z.getElement( 0, 2 ),   0.0 );
    EXPECT_EQ( Z.getElement( 0, 3 ),   0.0 );
    EXPECT_EQ( Z.getElement( 1, 0 ),   3.0 );
    EXPECT_EQ( Z.getElement( 1, 1 ), -12.0 );
    EXPECT_EQ( Z.getElement( 1, 2 ),  -1.0 );
    EXPECT_EQ( Z.getElement( 1, 3 ),   0.0 );
    EXPECT_EQ( Z.getElement( 2, 0 ),   1.0 );
    EXPECT_EQ( Z.getElement( 2, 1 ),   1.0 );
    EXPECT_EQ( Z.getElement( 2, 2 ),   4.0 );
    EXPECT_EQ( Z.getElement( 2, 3 ),   0.0 );
    EXPECT_EQ( Z.getElement( 3, 0 ),   0.0 );
    EXPECT_EQ( Z.getElement( 3, 1 ),  10.0 );
    EXPECT_EQ( Z.getElement( 3, 2 ),  34.0 );
    EXPECT_EQ( Z.getElement( 3, 3 ),  30.0 );
}

template< typename Matrix >
void test_CroutMethod10x10MatrixDecompose()
{
    using RealType = typename Matrix::RealType;

    const Matrix A( {
        { 1,     0,    0,    0,    0,    0,    0,    0,   0,    0 },
        { 2,  -215,    0,    0,    0,   21,    0,    0,   0,    0 },
        { 3,   114, -217,    0,    0,    0,   77,    0,  -1,    0 },
        { 4,   -89,  134, -203,    0,    0,    0,    0,   0,    0 },
        { 5,     0,  -80,  157,    2,    0,    0,    0,   0,    0 },
        { 0,    94, -150,  138, -142, -188,    0,    0,   0,    0 },
        { 7,  -170,    0, -109,  173,  170,  -97,    0,   0,    0 },
        { 8,   176, -253,  177, -241,    0,  140, -184,   0,    0 },
        { 9,     0, -230,    0, -254, -160,  116, -191, 235,    0 },
        { 10, -210,   49,    0,   23,   -9, -106,   -6,   0, -164 }
    } );

    Matrix L, U;
    L.setLike( A );
    U.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, L, U, BLOCK_SIZE );

    std::vector< RealType > lRowMajor = {
        1,     0,    0,    0,    0,    0,             0,             0,   0,             0,
        2,  -215,    0,    0,    0,    0,             0,             0,   0,             0,
        3,   114, -217,    0,    0,    0,             0,             0,   0,             0,
        4,   -89,  134, -203,    0,    0,             0,             0,   0,             0,
        5,     0,  -80,  157,    2,    0,             0,             0,   0,             0,
        0,    94, -150,  138, -142, -578.98727859,    0,             0,   0,             0,
        7,  -170,    0, -109,  173,  631.01829238, -221.7951865,     0,   0,             0,
        8,   176, -253,  177, -241, -661.37575564,  445.97373464, -184,   0,             0,
        9,     0, -230,    0, -254, -871.61931049,  234.55508466, -191, 235.91631069,    0,
        10, -210,   49,    0,   23,   36.37198918, -148.96705449,   -6,  -0.50721853, -164
     };

    std::vector< RealType > uRowMajor =  {
        1, 0, 0, 0, 0,  0,           0,          0,  0,          0,
        0, 1, 0, 0, 0, -0.09767442,  0,          0,  0,          0,
        0, 0, 1, 0, 0, -0.05131283, -0.35483871, 0,  0.00460829, 0,
        0, 0, 0, 1, 0,  0.00895125, -0.23422851, 0,  0.00304193, 0,
        0, 0, 0, 0, 1, -2.75518646,  4.19338948, 0, -0.0544596,  0,
        0, 0, 0, 0, 0,  1,          -0.99235174, 0,  0.01288768, 0,
        0, 0, 0, 0, 0,  0,           1,          0, -0.00730727, 0,
        0, 0, 0, 0, 0,  0,           0,          1,  0.00388498, 0,
        0, 0, 0, 0, 0,  0,           0,          0,  1,          0,
        0, 0, 0, 0, 0,  0,           0,          0,  0,          1
     };

    verifyDecompositionResult( L, U, lRowMajor, uRowMajor );

    Matrix Z;
    Z.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, Z, BLOCK_SIZE );

    std::vector< RealType > zRowMajor = {
        1,     0,    0,    0,    0,    0,             0,             0,   0,             0,
        2,  -215,    0,    0,    0,   -0.09767442,    0,             0,   0,             0,
        3,   114, -217,    0,    0,   -0.05131283,   -0.35483871,    0,   0.00460829,    0,
        4,   -89,  134, -203,    0,    0.00895125,   -0.23422851,    0,   0.00304193,    0,
        5,     0,  -80,  157,    2,   -2.75518646,    4.19338948,    0,  -0.0544596,     0,
        0,    94, -150,  138, -142, -578.98727859,   -0.99235174,    0,   0.01288768,    0,
        7,  -170,    0, -109,  173,  631.01829238, -221.7951865,     0,  -0.00730727,    0,
        8,   176, -253,  177, -241, -661.37575564,  445.97373464, -184,   0.00388498,    0,
        9,     0, -230,    0, -254, -871.61931049,  234.55508466, -191, 235.91631069,    0,
        10, -210,   49,    0,   23,   36.37198918, -148.96705449,   -6,  -0.50721853, -164
     };

    verifyDecompositionResult( Z, zRowMajor );
}

template< typename Matrix >
void test_CroutMethod19x19MatrixDecompose()
{
    using RealType = typename Matrix::RealType;

    Matrix A( 19, 19 );

    Decomposition::SampleMatrices< Matrix >::LF10::loadMatrixA( A );

    Matrix L, U;
    L.setLike( A );
    U.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, L, U, BLOCK_SIZE );

    std::vector< RealType > lRowMajor, uRowMajor;
    Decomposition::SampleMatrices< Matrix >::LF10::loadMatrixLRowMajor( lRowMajor );
    Decomposition::SampleMatrices< Matrix >::LF10::loadMatrixURowMajor( uRowMajor );

    verifyDecompositionResult( L, U, lRowMajor, uRowMajor );

    Matrix Z;
    Z.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, Z, BLOCK_SIZE );

    std::vector< RealType > zRowMajor;
    Decomposition::SampleMatrices< Matrix >::LF10::loadMatrixZRowMajor( zRowMajor );

    verifyDecompositionResult( Z, zRowMajor );
}

template< typename Matrix >
void test_CroutMethod38x38MatrixDecompose()
{
    using RealType = typename Matrix::RealType;

    Matrix A( 38, 38 );

    Decomposition::SampleMatrices< Matrix >::cage5::loadMatrixA( A );

    Matrix L, U;
    L.setLike( A );
    U.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, L, U, BLOCK_SIZE );

    std::vector< RealType > lRowMajor, uRowMajor;
    Decomposition::SampleMatrices< Matrix >::cage5::loadMatrixLRowMajor( lRowMajor );
    Decomposition::SampleMatrices< Matrix >::cage5::loadMatrixURowMajor( uRowMajor );

    verifyDecompositionResult( L, U, lRowMajor, uRowMajor, 0.08 );

    Matrix Z;
    Z.setLike( A );

    Decomposition::LU::CroutMethodIterative::decompose( A, Z, BLOCK_SIZE );

    std::vector< RealType > zRowMajor;
    Decomposition::SampleMatrices< Matrix >::cage5::loadMatrixZRowMajor( zRowMajor );

    verifyDecompositionResult( Z, zRowMajor, 0.08 );
}

template< typename Matrix >
void test_CroutMethod10x10DecomposeSingularMatrixShouldFail()
{
    const Matrix A( {
        {0,    0,    0,    0,    0,    0,    0,    0,    0,    0},
        {0, -215,    0,    0,    0,    0,    0,    0,    0,    0},
        {0,  114, -217,    0,    0,    0,    0,    0,   -1,    0},
        {0,  -89,  134, -203,    0,    0,    0,    0,    0,    0},
        {0,  -77,   77,  -80,  157,    0,    0,    0,    0,    0},
        {0,  94,  -150,  138, -142, -188,    0,    0,    0,    0},
        {0, -170,   57, -109,  173,  170,  -97,    0,    0,    0},
        {0,  176, -253,  177, -241, -120,  140, -184,    0,    0},
        {0,  117, -230,  180, -254, -160,  116, -191,  235,    0},
        {0, -210,   49, -174,   23,   -9, -106,   -6, -132, -164}
    } );

    Matrix L, U;
    L.setLike( A );
    U.setLike( A );

    std::ofstream output( "assertion_error_output_LUIterative.txt" );
    std::streambuf* p_cerrbuffer = std::cerr.rdbuf();
    std::cerr.rdbuf( output.rdbuf() );

    try
    {
        Decomposition::LU::CroutMethodIterative::decompose( A, L, U, BLOCK_SIZE );
    }
    catch( ... )
    {
        std::cerr.rdbuf( p_cerrbuffer );
        SUCCEED();
        return;
    }
    FAIL() << "Decomposition of Matrix A should have failed.";
}

template< typename Matrix >
class ILUCroutMethodTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

using MatrixTypes = ::testing::Types
<
    TNL::Matrices::DenseMatrix< float,  TNL::Devices::Host, short >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, short >,
    TNL::Matrices::DenseMatrix< float,  TNL::Devices::Host, int   >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int   >,
    TNL::Matrices::DenseMatrix< float,  TNL::Devices::Host, long  >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, long  >
#ifdef HAVE_CUDA
    ,TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, short, TNL::Algorithms::Segments::RowMajorOrder >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, short, TNL::Algorithms::Segments::RowMajorOrder >,
    TNL::Matrices::DenseMatrix< float,  TNL::Devices::Cuda, int,   TNL::Algorithms::Segments::RowMajorOrder >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int,   TNL::Algorithms::Segments::RowMajorOrder >,
    TNL::Matrices::DenseMatrix< float,  TNL::Devices::Cuda, long,  TNL::Algorithms::Segments::RowMajorOrder >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, long,  TNL::Algorithms::Segments::RowMajorOrder >
#endif
>;

TYPED_TEST_SUITE( ILUCroutMethodTest, MatrixTypes );


TYPED_TEST( ILUCroutMethodTest, croutMethodSanityTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_CroutMethodSanity< MatrixType >();
}

TYPED_TEST( ILUCroutMethodTest, croutMethod2x2MatrixDecomposeTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_CroutMethod2x2MatrixDecompose< MatrixType >();
}

TYPED_TEST( ILUCroutMethodTest, croutMethod3x3MatrixDecomposeTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_CroutMethod3x3MatrixDecompose< MatrixType >();
}

TYPED_TEST( ILUCroutMethodTest, croutMethod4x4SemiSparseMatrixDecomposeTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_CroutMethod4x4SemiSparseMatrixDecompose< MatrixType >();
}

TYPED_TEST( ILUCroutMethodTest, CroutMethod10x10MatrixDecomposeTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_CroutMethod10x10MatrixDecompose< MatrixType >();
}

TYPED_TEST( ILUCroutMethodTest, CroutMethod19x19MatrixDecomposeTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_CroutMethod19x19MatrixDecompose< MatrixType >();
}

TYPED_TEST( ILUCroutMethodTest, CroutMethod38x38MatrixDecomposeTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_CroutMethod38x38MatrixDecompose< MatrixType >();
}

TYPED_TEST( ILUCroutMethodTest, DISABLED_CroutMethod10x10DecomposeSingularMatrixShouldFailTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_CroutMethod10x10DecomposeSingularMatrixShouldFail< MatrixType >();
}

#endif // HAVE_GTEST
