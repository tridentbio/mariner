# Mariner Webapp

This repository holds the frontend of the mariner system.

## Requirements

- Node v16 and npm

## Building

Build artifacts are produced into the `build/` folder after running:

```console
npm run build
```

## Installation

This repository holds the frontend of the Mariner system, and it needs the [backend]() to be running.
So if you haven't, go there and follow the installation instructions.

```console
npm run install
```

Once installed, you can run it with:

```console
npm run start
```

## pre-commit

To get pre-commit hooks working run the command

`npm run prepare`

## RTK Code Generation

Code generation of Redux Toolkit APIs and types is possible for all APIs that export an Open API schema, such as those made with FastAPI like Mariner's. It is configured in the `openapi-config.ts` script and can be executed only if the API endpoint in the `config.schemaFile` is accessible from where the script runs.

This is also how we generate the typings of new layers and featurizers.

### Adding layers and featurizers

First you should make sure the added component was added to the exported Union type of all layer/featurizer components. Otherwise, the type changes won't be detected.

1. Run the following command to update the folders with generated code. It's advisable to commit before running the command

```console
npm run generate:rtkapi
```

2. Check and fix for typescript errors caused by possible API changes.

```console
npx tsc
```

3. Add type alias for new components in `src/interfaces/model-editor.ts`

4. Add a visit method to the `src/model-compiler/src/implementation/validation/visitors/ComponentVisitor.ts` as well a case for the `src/model-compiler/src/implementation/validation/Acceptor.ts`

## Contributing

### Code style

- `npm run prettier:fix`
- `npm run eslint:fix`

### Running tests

- `npm run cypress:open`

- `npm run cypress:test`

### Architecture

- `src/app`:

  - `rtk/generated`: Holds the Redux Toolkit generated files for the API (created by openapi-config.ts). We should avoid using the generated files directly. It's better to prematurely extend it the right way, and use the extended object (in `src/app/rtk/`). Merging 2 api objects with a repeated endpoint is a common mistake. When adding an endpoint to the api make sure it's not already being added through code generation, and if you must you can filter the endpoints in `openapi-config.ts`
  - `types`: Holds the typescript types for the API. The types here used are complemented by the generated above mentioned.
  - `websocket`: Holds the scripts for the websocket connection.
  - `api.ts`: The API client configuration (configures request and response interceptors). The api is imported by the generated RTK API files so the APIs are merged.
    Although not recommended, if 2 APIs have the same endpoint, the one from the files will be used.
  - `hooks.ts`: React redux hooks with the types configured.
  - `notifications.tsx`: A Context and react hook for showing notifications.
  - `store.ts`: The Redux store configuration.

- `src/components/`: This folder organizes the components in order to improve reusability and UI homogeneity. We're still in the middle of the process of organizing the components, so there are some components that are not in the right place yet. The current structure is:

  - `atoms/`: Basic UI elements that are not composed of other components. Examples: buttons, inputs, etc.
  - `molecules/`: UI elements composed of other components. Examples: cards, modals, etc.
  - `organisms/`: A complex component composed of other components. Examples: top bar, model editor, etc.
  - `templates/` Layout components that are used to organize the page. Examples: dashboard, model builder, etc.

- `src/features/`: This folder organizes the features of the application. Each feature has it's own pages, that must be referenced in `src/hooks/useAppNavigation.tsx` to show in the appropriate menu and path.

- `model-compiler/src`: This folder contains the code for the model-compiler, that validates if a given model spec is valid, pointing to possible errors or miss uses that would cause the experiment to be reattempted. It is organized as follows:
  - `implementation/`: Implementation of the model-compiler. See model-compiler's README for more information.
  - `interfaces/torch-model-editor.ts`: This single file contains all necessary types of the implementation.

---

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!
