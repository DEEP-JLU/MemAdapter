# Generation organization

The implementation is shared so every memory system uses the same generation and logging contract. `systems/` is the public experiment index: each system records the supported datasets, model families, and the two retained ablations. The reusable implementation is in `methods/memadapter/`; baseline and intervention launchers are in `methods/`.

No run outputs belong here. Configure credentials and endpoints only through the ignored local environment described in `../config/models.example.env`.
