Python3 is required. Either use `python3` and `pip3` variants, or activate a Python virtual environment.

# gym-roboy
pip install -e .

# Run tests
```bash
python -m pytest
```
# TODO: combine Baris fix (q reset is working -> branch deepRoboy-feature) with Simon's fix (qdot is not always zero -> rikscha_devel) into *ONE* working repository
# correct test_msj_ros_bridge_proxy_step where np.abs(x - y) == 0 is alowed

# TODO: eliminate code replication between test_msj_env.py and test_integration.py