import subprocess
import os
import glob

def run_kraken2(input_fastq, db_path, output_dir, threads=4):
    """Runs Kraken2 for taxonomic classification."""
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "kraken_report.txt")
    output_file = os.path.join(output_dir, "kraken_output.txt")
    
    cmd = [
        "kraken2", "--db", db_path, "--threads", str(threads),
        "--report", report_file, "--output", output_file, input_fastq
    ]
    print(f"🚀 Running Kraken2: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_prokka(input_fasta, output_dir, prefix="annotated", threads=4):
    """Runs Prokka for genome annotation."""
    cmd = [
        "prokka", "--outdir", output_dir, "--prefix", prefix,
        "--cpus", str(threads), "--force", input_fasta
    ]
    print(f"🚀 Running Prokka: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_roary(gff_dir, output_dir, threads=4):
    """Runs Roary for pangenome analysis."""
    # Gather all GFF files in the directory
    gff_files = glob.glob(os.path.join(gff_dir, "*.gff"))
    
    if not gff_files:
        raise FileNotFoundError(f"No GFF files found in {gff_dir}")
        
    cmd = [
        "roary", "-f", output_dir, "-e", "-n", "-v",
        "-p", str(threads)
    ] + gff_files
    
    print(f"🚀 Running Roary with {len(gff_files)} genomes...")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    print("Genomic processing wrappers loaded and ready.")
    # Example: run_prokka("isolate_01.fasta", "prokka_out")

