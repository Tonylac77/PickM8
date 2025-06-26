"""
Pytest-based test for the _create_complex_with_biopython() function.
Tests the function using the 1fvv ligand and protein files from test_data directory.
Output files are preserved in test_outputs directory for examination.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

import pytest
from rdkit import Chem

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.interactions import create_complex_with_biopython


class TestCreateComplexWithBiopython:
    """Test class for create_complex_with_biopython function."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment before each test."""
        # Define paths
        self.project_root = Path(__file__).parent.parent
        self.test_data_dir = self.project_root / "test_data"
        self.test_outputs_dir = self.project_root / "test_outputs"
        
        # Create test outputs directory if it doesn't exist
        self.test_outputs_dir.mkdir(exist_ok=True)
        
        # Define test file paths
        self.protein_file = self.test_data_dir / "1fvv_p.pdb"
        self.ligand_file = self.test_data_dir / "1fvv_l.sdf"
        
        # Create timestamp for unique output files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store output files created during tests for cleanup or examination
        self.created_files = []

    def test_input_files_exist(self):
        """Verify that the required input files exist."""
        assert self.protein_file.exists(), f"Protein file not found: {self.protein_file}"
        assert self.ligand_file.exists(), f"Ligand file not found: {self.ligand_file}"
        
        print(f"\n✓ Found protein file: {self.protein_file}")
        print(f"✓ Found ligand file: {self.ligand_file}")

    def test_load_ligand_from_sdf(self):
        """Test that we can load the ligand molecule from SDF file."""
        supplier = Chem.SDMolSupplier(str(self.ligand_file))
        ligand_mol = None
        
        for mol in supplier:
            if mol is not None:
                ligand_mol = mol
                break
        
        assert ligand_mol is not None, "Could not load ligand molecule from SDF file"
        assert ligand_mol.GetNumAtoms() > 0, "Ligand molecule has no atoms"
        
        print(f"\n✓ Loaded ligand molecule with {ligand_mol.GetNumAtoms()} atoms")

    def test_create_complex_with_biopython_basic(self):
        """
        Test the create_complex_with_biopython function with basic parameters.
        
        This test:
        1. Loads the protein and ligand from test_data
        2. Calls create_complex_with_biopython
        3. Verifies that a complex PDB file is created
        4. Checks basic properties of the output file
        5. Preserves the output file for examination
        """
        # Load ligand from SDF file
        supplier = Chem.SDMolSupplier(str(self.ligand_file))
        ligand_mol = None
        
        for mol in supplier:
            if mol is not None:
                ligand_mol = mol
                break
        
        assert ligand_mol is not None, "Could not load ligand molecule from SDF file"
        
        # Test the function
        ligand_name = "LIG"
        complex_path = create_complex_with_biopython(
            str(self.protein_file), 
            ligand_mol, 
            ligand_name
        )
        
        # Verify the function succeeded
        assert complex_path is not None, "Function returned None - complex creation failed"
        assert os.path.exists(complex_path), "Complex file does not exist"
        
        # Check file size
        file_size = os.path.getsize(complex_path)
        assert file_size > 0, "Complex file is empty"
        
        # Read and validate content
        with open(complex_path, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
        
        # Basic content checks
        atom_lines = [line for line in lines if line.startswith(('ATOM', 'HETATM'))]
        ligand_lines = [line for line in lines if line.startswith('HETATM')]
        protein_lines = [line for line in lines if line.startswith('ATOM')]
        
        assert len(atom_lines) > 0, "No ATOM/HETATM lines found in complex file"
        assert len(protein_lines) > 0, "No protein ATOM lines found in complex file"
        
        # Check for proper PDB format
        has_end = any(line.startswith('END') for line in lines)
        assert has_end, "Complex file missing END line"
        
        # Save the output to test_outputs directory for examination
        output_filename = f"complex_basic_{self.timestamp}.pdb"
        permanent_output_path = self.test_outputs_dir / output_filename
        
        # Copy the temporary file to permanent location
        shutil.copy2(complex_path, permanent_output_path)
        self.created_files.append(permanent_output_path)
        
        print(f"\n✅ Complex created successfully!")
        print(f"  📁 Temporary file: {complex_path}")
        print(f"  💾 Saved copy: {permanent_output_path}")
        print(f"  📊 File size: {file_size} bytes")
        print(f"  🧬 Total ATOM/HETATM lines: {len(atom_lines)}")
        print(f"  🦠 Protein ATOM lines: {len(protein_lines)}")
        print(f"  💊 Ligand HETATM lines: {len(ligand_lines)}")
        
        # Note: We don't delete the temporary file here so you can examine it

    def test_create_complex_with_custom_ligand_name(self):
        """Test the function with a custom ligand name."""
        # Load ligand
        supplier = Chem.SDMolSupplier(str(self.ligand_file))
        ligand_mol = next(mol for mol in supplier if mol is not None)
        
        assert ligand_mol is not None, "Could not load ligand molecule"
        
        # Test with custom ligand name
        custom_ligand_name = "TEST_LIG"
        complex_path = create_complex_with_biopython(
            str(self.protein_file), 
            ligand_mol, 
            custom_ligand_name
        )
        
        assert complex_path is not None, "Function returned None with custom ligand name"
        assert os.path.exists(complex_path), "Complex file does not exist with custom ligand name"
        
        # Check file content
        with open(complex_path, 'r') as f:
            content = f.read()
        
        # Save this output too
        output_filename = f"complex_custom_name_{self.timestamp}.pdb"
        permanent_output_path = self.test_outputs_dir / output_filename
        
        shutil.copy2(complex_path, permanent_output_path)
        self.created_files.append(permanent_output_path)
        
        print(f"\n✅ Custom ligand name test completed!")
        print(f"  💾 Saved copy: {permanent_output_path}")
        print(f"  🏷️  Ligand name: {custom_ligand_name}")

    def test_create_complex_multiple_calls(self):
        """Test that multiple calls to the function work correctly."""
        # Load ligand
        supplier = Chem.SDMolSupplier(str(self.ligand_file))
        ligand_mol = next(mol for mol in supplier if mol is not None)
        
        # Make multiple calls with different ligand names
        ligand_names = ["LIG1", "LIG2", "DRUG"]
        created_complexes = []
        
        for i, ligand_name in enumerate(ligand_names):
            complex_path = create_complex_with_biopython(
                str(self.protein_file), 
                ligand_mol, 
                ligand_name
            )
            
            assert complex_path is not None, f"Function failed for ligand name {ligand_name}"
            assert os.path.exists(complex_path), f"Complex file missing for {ligand_name}"
            
            # Save each output
            output_filename = f"complex_multi_{i}_{ligand_name}_{self.timestamp}.pdb"
            permanent_output_path = self.test_outputs_dir / output_filename
            
            shutil.copy2(complex_path, permanent_output_path)
            self.created_files.append(permanent_output_path)
            created_complexes.append((ligand_name, permanent_output_path))
        
        print(f"\n✅ Multiple calls test completed!")
        for ligand_name, path in created_complexes:
            print(f"  📄 {ligand_name}: {path}")

    def test_error_handling_invalid_protein(self):
        """Test that function handles invalid protein file gracefully."""
        # Load a valid ligand
        supplier = Chem.SDMolSupplier(str(self.ligand_file))
        ligand_mol = next(mol for mol in supplier if mol is not None)
        
        # Test with non-existent protein file
        # The function should return None instead of raising an exception
        result = create_complex_with_biopython(
            "/non/existent/protein.pdb", 
            ligand_mol, 
            "LIG"
        )
        
        assert result is None, "Function should return None for invalid protein file"
        
        print(f"\n✅ Error handling test passed - invalid protein file correctly rejected")

    def test_error_handling_invalid_ligand(self):
        """Test that function handles invalid ligand gracefully."""
        # Test with None ligand
        with pytest.raises(Exception):  # Could be AttributeError, TypeError, or other exceptions
            create_complex_with_biopython(
                str(self.protein_file), 
                None, 
                "LIG"
            )
        
        print(f"\n✅ Error handling test passed - None ligand correctly rejected")

    def teardown_method(self):
        """Optional teardown - prints info about created files."""
        if self.created_files:
            print(f"\n📁 Created {len(self.created_files)} output files in {self.test_outputs_dir}:")
            for file_path in self.created_files:
                if file_path.exists():
                    size = file_path.stat().st_size
                    print(f"  📄 {file_path.name} ({size} bytes)")


if __name__ == "__main__":
    # Allow running the test directly with verbose output
    pytest.main([__file__, "-v", "-s"])
