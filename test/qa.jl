# Code quality: Aqua (ambiguities, unbound args, stale deps, compat, piracy),
# JET (static analysis of the package), ExplicitImports.

import Aqua
import ExplicitImports
import JET

@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(L2ODLL)
    end

    @testset "ExplicitImports" begin
        @test ExplicitImports.check_no_implicit_imports(L2ODLL) === nothing
        @test ExplicitImports.check_no_stale_explicit_imports(L2ODLL) === nothing
        @test ExplicitImports.check_all_explicit_imports_via_owners(L2ODLL) === nothing
        # `DI.AbstractADType` is re-exported by DifferentiationInterface from ADTypes;
        # accessing it through DI avoids a direct dependency on ADTypes.
        @test ExplicitImports.check_all_qualified_accesses_via_owners(L2ODLL; ignore = (:AbstractADType,)) === nothing
    end

    @testset "JET" begin
        JET.test_package(L2ODLL; target_modules = (L2ODLL,))
    end
end
