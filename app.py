import os
from flask import Flask
from flask import request
from flask import jsonify
import subprocess
from subprocess import check_output
from flask_cors import CORS

app=Flask(__name__)
CORS(app)


def call_sh_script(script):
    script_dir=os.path.join(os.getcwd(),"scripts",script)
    read_script=check_output([script_dir]).decode('utf-8')
    print(read_script)
    return read_script

@app.get("/")
def index():
    return "Hello custom-diffusion"

@app.get("/run_sh/<script_name>")
def run_sh(script_name):
    try:
        if script_name=="finetune_gen.sh":
            print("finetune_gen.sh is running")
            call_sh_script(script_name)
            return jsonify({"message":"finetune_gen.sh script_name ran successfully"}), 200
        elif script_name=="finetune_joint.sh":
            print("finetune_joint.sh is running")
            call_sh_script(script_name)
            return jsonify({"message":"finetune_joint.sh script_name ran successfully"}), 200
        elif script_name=="finetune_real.sh":
            print("finetune_real.sh is running")
            call_sh_script(script_name)
            return jsonify({"message":"finetune_real.sh script_name ran successfully"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)}), 500

@app.get("/sample")
def sample():
    try:
        user_params = request.args.to_dict()
        args=[]
        for name,val in user_params.items():
            args.append("--{}".format(name))
            args.append(val)
        subprocess.run(["python","sample.py"]+args)
        return jsonify({"message":"sample.py script is running"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)}), 500

@app.get("/train")
def train():
    try:
        user_params = request.args.to_dict()
        args=[]
        for name,val in user_params.items():
            args.append("--{}".format(name))
            args.append(val)
        subprocess.run(["python","train.py"]+args)
        return jsonify({"message":"train.py script is running"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error":str(e)}), 500

if __name__=="__main__":
    app.run(debug=True,host="localhost",port=8000)