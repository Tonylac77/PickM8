# Test Data Documentation

This directory contains real molecular data files used for integration testing in PickM8.

## Current Test Data Files

### 1. **1fvv_p.pdb** - Protein Structure
- **Source**: HIV Protease (PDB ID: 1FVV)
- **Format**: Protein Data Bank (PDB) format
- **Usage**: Used in protein-ligand interaction tests, complex creation tests, and pose quality analysis
- **Contains**: Complete protein structure with atomic coordinates

### 2. **1fvv_l.sdf** - Single Ligand Structure  
- **Source**: Ligand from HIV Protease complex (PDB ID: 1FVV)
- **Format**: Structure Data File (SDF) format
- **Usage**: Used in SDF loading tests, single molecule processing tests
- **Contains**: Single ligand molecule with 3D coordinates and properties

### 3. **example_poses_1fvv.sdf** - Multiple Ligand Poses
- **Source**: Multiple conformations/poses of the 1FVV ligand  
- **Format**: Multi-molecule SDF file
- **Usage**: Used in multi-molecule processing tests, pose quality comparison tests, batch processing tests
- **Contains**: Multiple ligand conformations (26 atoms each) with different 3D coordinates

## Test Coverage

### Current Tests Using Real Data

**High Priority (Implemented):**
- ✅ **test_data_processing.py**: SDF loading, property detection, score processing
- ✅ **test_interactions.py**: PLIP/ProLIF protein-ligand interaction calculations  
- ✅ **test_posecheck_analysis.py**: Pose quality analysis, clash detection, strain energy
- ✅ **test_create_complex_biopython.py**: Already used real data excellently

**Medium Priority (Could be enhanced):**
- ⚠️ **test_molecular_fingerprints.py**: Could use real molecules for fingerprint validation
- ⚠️ **test_active_learning_models.py**: Pure algorithmic tests, likely doesn't need real molecular data

## Recommended Additional Test Data Files

The following files would enhance testing coverage if added:

### 1. **small_molecules.sdf** - Simple Test Molecules
- **Purpose**: Fast unit testing with simple, well-characterized molecules
- **Contents**: 3-5 small molecules (caffeine, aspirin, etc.) with known properties
- **Usage**: Quick fingerprint validation, basic interaction tests
- **Size**: Small (~10-20 atoms per molecule)

### 2. **invalid_molecules.sdf** - Error Handling Test Data
- **Purpose**: Test error handling and edge cases
- **Contents**: Problematic molecular structures (invalid bonds, missing atoms, etc.)
- **Usage**: Robustness testing, error handling validation
- **Expected**: Some molecules should fail processing gracefully

### 3. **scored_molecules.sdf** - Molecules with Score Properties
- **Purpose**: Test score processing and active learning workflows
- **Contents**: Molecules with various numerical properties (IC50, binding affinity, etc.)
- **Usage**: Score column processing, ML model training tests
- **Properties**: Multiple numeric columns with different ranges and distributions

### 4. **large_protein.pdb** - Large Protein Complex
- **Purpose**: Performance testing and memory usage validation
- **Contents**: Large protein structure (>5000 atoms)
- **Usage**: Stress testing, performance benchmarks
- **Benefits**: Identify scalability issues

### 5. **multi_chain_complex.pdb** - Multi-Chain Complex
- **Purpose**: Test complex protein structures
- **Contents**: Protein complex with multiple chains and cofactors
- **Usage**: Advanced interaction analysis, complex processing
- **Benefits**: More realistic biological systems

### 6. **chiral_molecules.sdf** - Stereochemically Rich Molecules
- **Purpose**: Test chiral fingerprint calculations (MapChiral)
- **Contents**: Molecules with multiple chiral centers
- **Usage**: Stereochemical fingerprint validation
- **Benefits**: Test advanced fingerprint methods

## Usage Guidelines

### For Test Development

1. **Start with existing data**: Use current files for new test development
2. **Test both success and failure cases**: Include edge cases and error conditions
3. **Use appropriate file sizes**: Balance realism with test execution time
4. **Document expected outcomes**: Each test should have clear success criteria

### For CI/CD

1. **Include in version control**: All test data should be tracked
2. **Keep files small**: Large files slow down testing and repository cloning
3. **Test graceful degradation**: Tests should handle missing external tools (PLIP, ProLIF, PoseCheck)
4. **Use conditional skipping**: Skip tests when dependencies aren't available

### Test Performance Considerations

- **Real data tests are slower**: Integration tests take 10-100x longer than mocked tests
- **Conditional execution**: Use `@pytest.mark.skipif` for optional dependencies
- **Parallel execution**: Real data tests can often run in parallel
- **Resource cleanup**: Ensure temporary files are properly cleaned up

## Integration Test Strategy

### Current Approach (Implemented)
- **Dual testing**: Keep both mocked unit tests and real data integration tests
- **Graceful degradation**: Tests skip when tools aren't available
- **Progressive validation**: Start with structure validation, then semantic validation
- **Error resilience**: Tests handle failures and provide useful error messages

### Benefits of Real Data Testing
- **Catches integration bugs**: Issues that mocks miss
- **Validates complete workflows**: End-to-end processing verification  
- **Real-world edge cases**: Discovers issues with actual molecular data
- **Performance insights**: Understanding of actual computational costs
- **Tool compatibility**: Validates integration with external libraries

### Test Maintenance
- **Review regularly**: Ensure tests remain relevant as code evolves
- **Update test data**: Add new molecular systems as needed
- **Monitor performance**: Track test execution times
- **Update documentation**: Keep this file current with test changes