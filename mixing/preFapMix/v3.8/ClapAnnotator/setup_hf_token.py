#!/usr/bin/env python
"""
Script to set up Hugging Face token either via .env file or huggingface-cli login.
This is a helper script for setting up the CLAP Annotator environment.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

def create_env_file(token: str, force: bool = False) -> bool:
    """Create a .env file with the Hugging Face token.
    
    Args:
        token: The Hugging Face token to use
        force: Whether to overwrite an existing .env file
        
    Returns:
        True if the file was created or updated, False otherwise
    """
    env_path = Path('.env')
    
    if env_path.exists() and not force:
        log.warning(f".env file already exists at {env_path.absolute()}")
        log.info("Use --force to overwrite it")
        return False
    
    try:
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(f'HF_TOKEN="{token}"\n')
            f.write('LOG_LEVEL="INFO"\n')
        
        log.info(f"Created .env file at {env_path.absolute()}")
        return True
    except Exception as e:
        log.error(f"Failed to create .env file: {e}")
        return False

def login_with_cli(token: str = None) -> bool:
    """Log in to Hugging Face using huggingface-cli.
    
    Args:
        token: Optional token to use for login
        
    Returns:
        True if login was successful, False otherwise
    """
    try:
        # Check if huggingface_hub is installed
        try:
            import huggingface_hub
            log.info(f"huggingface_hub version: {huggingface_hub.__version__}")
        except ImportError:
            log.error("huggingface_hub not installed. Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.19.0"])
            log.info("huggingface_hub installed successfully")
        
        if token:
            # Login with provided token
            log.info("Logging in to Hugging Face with provided token...")
            result = subprocess.run(
                ["huggingface-cli", "login", "--token", token],
                capture_output=True,
                text=True
            )
        else:
            # Interactive login
            log.info("Starting interactive login to Hugging Face...")
            log.info("You will be prompted to enter your token.")
            log.info("You can get a token from https://huggingface.co/settings/tokens")
            result = subprocess.run(["huggingface-cli", "login"])
            
        if result.returncode == 0:
            log.info("Successfully logged in to Hugging Face")
            return True
        else:
            log.error(f"Failed to log in to Hugging Face: {result.stderr}")
            return False
    except Exception as e:
        log.error(f"Error during Hugging Face login: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Set up Hugging Face token for CLAP Annotator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Token options
    token_group = parser.add_mutually_exclusive_group()
    token_group.add_argument(
        "--token",
        type=str,
        help="Hugging Face token to use"
    )
    token_group.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive login with huggingface-cli"
    )
    
    # Method options
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--env-file",
        action="store_true",
        help="Create a .env file with the token"
    )
    method_group.add_argument(
        "--cli-login",
        action="store_true",
        help="Use huggingface-cli login"
    )
    
    # Other options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing .env file"
    )
    
    args = parser.parse_args()
    
    # Default to CLI login if no method specified
    use_cli_login = args.cli_login or not args.env_file
    
    # If no token method specified, default to interactive if using CLI login
    if not args.token and not args.interactive:
        args.interactive = use_cli_login
    
    # Get token if needed
    token = args.token
    
    # Perform setup
    if use_cli_login:
        success = login_with_cli(token)
    else:
        if not token:
            log.error("Token is required when using --env-file")
            parser.print_help()
            return 1
        success = create_env_file(token, args.force)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 