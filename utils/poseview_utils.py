import requests
import tempfile
import time
import base64
import json
import logging
from pathlib import Path
from rdkit import Chem

logger = logging.getLogger(__name__)

class PoseViewAPI:
    """Wrapper for PoseView API to generate 2D interaction diagrams"""
    
    def __init__(self):
        self.base_url = "https://proteins.plus/api/poseview_rest"
        self.max_wait_time = 60  # Maximum time to wait for job completion (seconds)
        self.poll_interval = 2   # How often to check job status (seconds)
    
    def create_job_with_files(self, protein_content, ligand_mol_block, ligand_name="ligand"):
        """
        Create a PoseView job by uploading protein and ligand files
        
        Args:
            protein_content (str): PDB file content
            ligand_mol_block (str): SDF/MOL block content
            ligand_name (str): Name for the ligand
            
        Returns:
            dict: Job response with location or None if failed
        """
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as pdb_file:
                pdb_file.write(protein_content)
                pdb_path = pdb_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as sdf_file:
                sdf_file.write(ligand_mol_block)
                sdf_path = sdf_file.name
            
            # Prepare multipart form data
            files = {
                'protein': ('protein.pdb', open(pdb_path, 'rb'), 'chemical/x-pdb'),
                'ligand': ('ligand.sdf', open(sdf_path, 'rb'), 'chemical/x-mdl-sdfile')
            }
            
            data = {
                'ligandName': ligand_name
            }
            
            # Make request
            response = requests.post(
                f"{self.base_url}/upload",
                files=files,
                data=data,
                timeout=30
            )
            
            # Clean up files
            for file_obj in files.values():
                file_obj[1].close()
            Path(pdb_path).unlink(missing_ok=True)
            Path(sdf_path).unlink(missing_ok=True)
            
            if response.status_code in [200, 202]:
                return response.json()
            else:
                logger.error(f"PoseView job creation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create PoseView job: {str(e)}")
            return None
    
    def create_job_with_pdb_code(self, pdb_code, ligand_id):
        """
        Create a PoseView job using PDB code and ligand ID
        
        Args:
            pdb_code (str): 4-letter PDB code
            ligand_id (str): Ligand identifier in the PDB
            
        Returns:
            dict: Job response with location or None if failed
        """
        try:
            payload = {
                "poseview": {
                    "pdbCode": pdb_code,
                    "ligand": ligand_id
                }
            }
            
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code in [200, 202]:
                return response.json()
            else:
                logger.error(f"PoseView job creation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create PoseView job: {str(e)}")
            return None
    
    def get_job_status(self, job_id):
        """
        Get the status of a PoseView job
        
        Args:
            job_id (str): Job ID returned from job creation
            
        Returns:
            dict: Job status response or None if failed
        """
        try:
            response = requests.get(
                f"{self.base_url}/{job_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 202:
                return response.json()  # Still processing
            else:
                logger.error(f"Failed to get job status: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get job status: {str(e)}")
            return None
    
    def wait_for_completion(self, job_id):
        """
        Wait for a job to complete and return the results
        
        Args:
            job_id (str): Job ID to wait for
            
        Returns:
            dict: Job results with image data or None if failed/timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < self.max_wait_time:
            status = self.get_job_status(job_id)
            
            if status is None:
                return None
            
            if status.get('status_code') == 200:
                # Job completed successfully
                return status
            elif status.get('status_code') == 202:
                # Still processing, wait and try again
                time.sleep(self.poll_interval)
            else:
                # Error occurred
                logger.error(f"Job failed with status: {status}")
                return None
        
        # Timeout
        logger.warning(f"Job {job_id} timed out after {self.max_wait_time} seconds")
        return None
    
    def generate_interaction_diagram(self, protein_content, ligand_mol_block, ligand_name="ligand"):
        """
        Complete workflow to generate a 2D interaction diagram
        
        Args:
            protein_content (str): PDB file content
            ligand_mol_block (str): SDF/MOL block content
            ligand_name (str): Name for the ligand
            
        Returns:
            dict: Dictionary containing image data (PNG, SVG, PDF) or None if failed
        """
        try:
            # Create job
            job_response = self.create_job_with_files(protein_content, ligand_mol_block, ligand_name)
            
            if not job_response:
                return None
            
            # Extract job ID from location
            location = job_response.get('location', '')
            if not location:
                logger.error("No location provided in job response")
                return None
            
            # Extract job ID (last part of the location URL)
            job_id = location.split('/')[-1]
            
            if job_response.get('status_code') == 200:
                # Job already exists and is complete
                return self.get_job_status(job_id)
            else:
                # Wait for job completion
                return self.wait_for_completion(job_id)
                
        except Exception as e:
            logger.error(f"Failed to generate interaction diagram: {str(e)}")
            return None
    
    def get_png_image_data(self, job_result):
        """
        Extract PNG image data from job result
        
        Args:
            job_result (dict): Job result from PoseView API
            
        Returns:
            bytes: PNG image data or None if not available
        """
        try:
            png_data = job_result.get('result_png_picture', '')
            if png_data:
                # The API returns base64-encoded image data
                return base64.b64decode(png_data)
            return None
        except Exception as e:
            logger.error(f"Failed to decode PNG data: {str(e)}")
            return None
    
    def get_svg_image_data(self, job_result):
        """
        Extract SVG image data from job result
        
        Args:
            job_result (dict): Job result from PoseView API
            
        Returns:
            str: SVG image data or None if not available
        """
        try:
            svg_data = job_result.get('result_svg_picture', '')
            if svg_data:
                # SVG might be base64 encoded or direct text
                try:
                    return base64.b64decode(svg_data).decode('utf-8')
                except:
                    return svg_data  # Already decoded
            return None
        except Exception as e:
            logger.error(f"Failed to decode SVG data: {str(e)}")
            return None