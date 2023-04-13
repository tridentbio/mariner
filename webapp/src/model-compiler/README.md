# Mariner model compiler

This package aims to provide model building and make suggestions for the architecture being build

## Goals
- [x] Generate LayerType and FeaturizerType types using open api code generation.
    - First download the openapi.json schema from API.
    - Second, run `npx openapi-typescript ./openapi.json --output ./schema.ts`
- [ ] Implement the ModelEditor interface
    - [x] addComponent
    - [x] editComponent
    - [x] deleteComponent
    - [x] getSuggestions
        - [-] Shapes and Data Type visitor
            - [x] Input
            - [x] GCNConv
            - [x] Linear
            - [x] Sigmoid, Relu
            - [x] MoleculeFeaturizer
            - [-] Concat  ( not tested )
            - [ ] GlobalPooling
        - [ ] Layer and featurizers validatores
            - [x] GCNConv
            - [x] Linear
            - [x] MoleculeFeaturizer
            - [x] Sigmoid, Relu
            - [x] Concat
            - [ ] GlobalPooling
    - [x] applySuggestions
- [ ] Benchmark
- [ ] Test coverage and documentation


