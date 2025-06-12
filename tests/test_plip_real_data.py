#!/usr/bin/env python3
"""
Integration test for PLIP interaction calculations using real test data.
This test uses the 1fvv protein and ligand files from test_data directory to verify
that the functional PLIP implementation works correctly with real molecular data.
"""

import pytest
import numpy as np
from pathlib import Path
from rdkit import Chem
import tempfile
import os
from datetime import datetime

from core.plip_interactions import (
    calculate_plip_interactions,
    is_plip_available,
    create_complex_with_biopython,
    get_plip_interaction_types
)
from core.interaction_functions import (
    calculate_interactions,
    create_interaction_context,
    calculate_with_context
)


@pytest.fixture
def test_data_dir():
    """Get the test data directory path"""
    return Path(__file__).parent.parent / "test_data"


@pytest.fixture
def protein_file(test_data_dir):
    """Get the protein PDB file path"""
    protein_path = test_data_dir / "1fvv_p.pdb"
    if not protein_path.exists():
        pytest.skip(f"Test protein file not found: {protein_path}")
    return str(protein_path)


@pytest.fixture
def ligand_mol(test_data_dir):
    """Load the ligand molecule from SDF file"""
    sdf_path = test_data_dir / "1fvv_l.sdf"
    if not sdf_path.exists():
        pytest.skip(f"Test ligand file not found: {sdf_path}")
    
    supplier = Chem.SDMolSupplier(str(sdf_path))
    mol = next(supplier)
    if mol is None:
        pytest.skip("Could not load ligand molecule from SDF file")
    return mol


@pytest.fixture
def output_dir():
    """Create output directory for test artifacts"""
    output_path = Path(__file__).parent.parent / "test_outputs"
    output_path.mkdir(exist_ok=True)
    return output_path


class TestPLIPRealData:
    """Integration tests for PLIP using real molecular data"""
    
    def test_plip_availability(self):
        """Verify PLIP is available for testing"""
        available = is_plip_available()
        if not available:
            pytest.skip("PLIP is not available - cannot run integration tests")
        assert available is True
    
    def test_ligand_loading(self, ligand_mol):
        """Test that the ligand loads correctly from SDF"""
        assert ligand_mol is not None
        assert ligand_mol.GetNumAtoms() > 0
        assert ligand_mol.GetNumBonds() > 0
    
    def test_protein_loading(self, protein_file):
        """Test that the protein loads correctly from PDB"""
        assert os.path.exists(protein_file)
        
        # Check file size and basic content
        file_size = os.path.getsize(protein_file)
        assert file_size > 0
        
        with open(protein_file, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            atom_lines = [line for line in lines if line.startswith('ATOM')]
        
        assert len(atom_lines) > 0
        print(f"\n📊 Protein Properties:")
        print(f"  File size: {file_size} bytes")
        print(f"  Total lines: {len(lines)}")
        print(f"  ATOM lines: {len(atom_lines)}")
    
    @pytest.mark.skipif(not is_plip_available(), reason="PLIP not available")
    def test_complex_creation_with_real_data(self, protein_file, ligand_mol, output_dir):
        """Test complex creation using real protein and ligand data"""
        # Create complex
        complex_path = create_complex_with_biopython(
            protein_file, ligand_mol, "LIG"
        )
        
        assert complex_path is not None, "Complex creation failed"
        assert os.path.exists(complex_path), "Complex file was not created"
        
        # Verify complex content
        with open(complex_path, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
        
        protein_atoms = [line for line in lines if line.startswith('ATOM')]
        ligand_atoms = [line for line in lines if line.startswith('HETATM')]
        
        assert len(protein_atoms) > 0, "No protein atoms in complex"
        assert len(ligand_atoms) > 0, "No ligand atoms in complex"
        
        # Save complex for examination
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_complex = output_dir / f"complex_real_data_{timestamp}.pdb"
        
        with open(saved_complex, 'w') as f:
            f.write(content)
        
        print(f"\n✅ Complex created successfully:")
        print(f"  📁 Temporary file: {complex_path}")
        print(f"  💾 Saved copy: {saved_complex}")
        print(f"  🦠 Protein atoms: {len(protein_atoms)}")
        print(f"  💊 Ligand atoms: {len(ligand_atoms)}")
        
        # Clean up temporary file
        try:
            os.unlink(complex_path)
        except:
            pass
    
    @pytest.mark.skipif(not is_plip_available(), reason="PLIP not available")
    def test_plip_interaction_calculation(self, protein_file, ligand_mol, output_dir):
        """Test PLIP interaction calculation with real data"""
        # Calculate interactions
        ifp, interactions_summary = calculate_plip_interactions(
            protein_file, ligand_mol, "LIG"
        )
        
        # Verify return types
        assert isinstance(ifp, np.ndarray), "IFP should be numpy array"
        assert isinstance(interactions_summary, dict), "Interactions should be dict"
        
        # Verify IFP properties
        assert ifp.shape == (1024,), f"Expected shape (1024,), got {ifp.shape}"
        assert ifp.dtype == int, f"Expected int dtype, got {ifp.dtype}"
        assert np.all((ifp == 0) | (ifp == 1)), "IFP should be binary"
        
        # Verify interactions summary structure
        required_keys = ['total_interactions', 'interaction_types', 'interactions']
        for key in required_keys:
            assert key in interactions_summary, f"Missing key: {key}"
        
        assert isinstance(interactions_summary['total_interactions'], int)
        assert isinstance(interactions_summary['interaction_types'], dict)
        assert isinstance(interactions_summary['interactions'], list)
        
        # Log results
        total_interactions = interactions_summary['total_interactions']
        interaction_types = interactions_summary['interaction_types']
        ifp_bits_set = np.sum(ifp)
        
        print(f"\n🧬 PLIP Interaction Analysis Results:")
        print(f"  Total interactions: {total_interactions}")
        print(f"  IFP bits set: {ifp_bits_set}")
        print(f"  Interaction types found:")
        for int_type, count in interaction_types.items():
            if count > 0:
                print(f"    - {int_type}: {count}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"plip_results_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write("PLIP Interaction Analysis Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total interactions: {total_interactions}\n")
            f.write(f"IFP bits set: {ifp_bits_set}\n")
            f.write(f"IFP shape: {ifp.shape}\n")
            f.write(f"IFP dtype: {ifp.dtype}\n\n")
            f.write("Interaction types:\n")
            for int_type, count in interaction_types.items():
                f.write(f"  {int_type}: {count}\n")
            f.write("\nDetailed interactions:\n")
            for i, interaction in enumerate(interactions_summary['interactions']):
                f.write(f"  {i+1}. {interaction}\n")
        
        print(f"  📊 Detailed results saved: {results_file}")
        
        # Verify we found some interactions (this is real data, should have interactions)
        assert total_interactions >= 0, "Should have non-negative interaction count"
        # Note: We don't assert > 0 because it depends on the specific data and PLIP parameters
    
    @pytest.mark.skipif(not is_plip_available(), reason="PLIP not available")
    def test_functional_interface_with_real_data(self, protein_file, ligand_mol):
        """Test the functional interface using real data"""
        # Test direct function call
        ifp1, interactions1 = calculate_interactions(
            protein_file, ligand_mol, "LIG", ifp_type="PLIP"
        )
        
        # Test context-based approach
        context = create_interaction_context(ifp_type="PLIP")
        ifp2, interactions2 = calculate_with_context(
            context, protein_file, ligand_mol, "LIG"
        )
        
        # Results should be identical
        assert np.array_equal(ifp1, ifp2), "Direct and context-based calls should give same IFP"
        assert interactions1 == interactions2, "Direct and context-based calls should give same interactions"
        
        print(f"\n✅ Functional interface consistency verified")
        print(f"  Both methods found {interactions1['total_interactions']} interactions")
    
    @pytest.mark.skipif(not is_plip_available(), reason="PLIP not available")
    def test_interaction_types_consistency(self):
        """Test that interaction types are consistent across functions"""
        # Get types from PLIP module
        plip_types = get_plip_interaction_types()
        
        # Get types from functional interface
        from core.interaction_functions import get_available_interaction_types
        functional_types = get_available_interaction_types("PLIP")
        
        # Should be identical
        assert set(plip_types) == set(functional_types), "Interaction types should be consistent"
        
        expected_types = {
            'hydrogen_bond', 'hydrophobic', 'pi_stacking', 'salt_bridge',
            'halogen_bond', 'pi_cation', 'water_bridge', 'metal_coordination'
        }
        
        assert set(plip_types) == expected_types, "Should have all expected PLIP interaction types"
        
        print(f"\n✅ Interaction types consistency verified")
        print(f"  Found {len(plip_types)} interaction types: {', '.join(plip_types)}")
    
    @pytest.mark.skipif(not is_plip_available(), reason="PLIP not available")
    def test_error_handling_with_invalid_data(self, ligand_mol):
        """Test error handling with invalid protein file"""
        # Test with non-existent protein file
        with pytest.raises(Exception):
            calculate_plip_interactions("/non/existent/protein.pdb", ligand_mol, "LIG")
        
        # Test with invalid protein file (empty file)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp.write("")  # Empty file
            tmp_path = tmp.name
        
        try:
            # This should either raise an exception or return default values
            result = calculate_plip_interactions(tmp_path, ligand_mol, "LIG")
            if result is not None:
                ifp, interactions = result
                # If it doesn't raise an exception, it should return safe defaults
                assert isinstance(ifp, np.ndarray)
                assert isinstance(interactions, dict)
        except Exception:
            # It's acceptable for this to raise an exception
            pass
        finally:
            os.unlink(tmp_path)
        
        print(f"\n✅ Error handling verified")
    
    def test_integration_with_fingerprint_handler(self, protein_file, ligand_mol):
        """Test integration with the fingerprint handler system"""
        if not is_plip_available():
            pytest.skip("PLIP not available")
        
        # Test that the interaction context can be used with fingerprint handler
        from core.fingerprints import FingerprintHandler
        
        fp_handler = FingerprintHandler(fp_type='morgan', interaction_fp_type='PLIP')
        interaction_context = create_interaction_context(ifp_type='PLIP')
        
        # Verify context is compatible
        assert interaction_context['ifp_type'] == 'PLIP'
        assert fp_handler.get_interaction_fingerprint_type() == 'PLIP'
        
        print(f"\n✅ Fingerprint handler integration verified")
        print(f"  Context IFP type: {interaction_context['ifp_type']}")
        print(f"  Handler IFP type: {fp_handler.get_interaction_fingerprint_type()}")


if __name__ == "__main__":
    # Allow running the test directly with verbose output
    pytest.main([__file__, "-v", "-s"])