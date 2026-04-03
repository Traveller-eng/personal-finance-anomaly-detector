import bcrypt
import getpass

def main():
    print("=" * 50)
    print("PFAD Configuration Hash Generator")
    print("=" * 50)
    print("\nThis utility creates a secure bcrypt hash for your config.yaml file.")
    
    password = getpass.getpass("\nEnter the desired password: ")
    confirm_password = getpass.getpass("Confirm password: ")
    
    if password != confirm_password:
        print("\n[!] Error: Passwords do not match.")
        return

    # Generate bcrypt hash compatible with streamlit-authenticator
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode()
    
    print("\n" + "=" * 50)
    print("SUCCESS! Copy the string below into your config.yaml")
    print("=" * 50)
    print(f'\npassword: "{hashed}"\n')
    print("=" * 50)

if __name__ == "__main__":
    main()
