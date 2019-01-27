## Unit tests
To run the tests, you need to source your ROS2_ROBOY_WS first.
For unit tests, no running CARDSflow simulation is necessary.
```bash
python3.5 -m pytest --disable-warnings -rs -v
```
* `--disable-warnings ` to mute Tensorflow warnings.
* `-rs` shows a report of skipped tests
* `-v` for verbose
### Integration tests
If you want to run the integration tests, you will need a running CARDSflow simulation.
In that case:
```bash
python3.5 -m pytest --disable-warnings -rs -v --run-integration
```