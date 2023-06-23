# Mariner Webapp

This repository hols the frontend of the mariner system.

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

`npm run install`

## pre-commit

To get pre-commit hooks working run the command

`npm run prepare`

## Adding layers and featurizers

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
