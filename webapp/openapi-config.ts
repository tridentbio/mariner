import type { ConfigFile } from '@rtk-query/codegen-openapi';
const config: ConfigFile = {
  schemaFile: 'http://localhost:8888/openapi.json',
  apiFile: './src/app/rtk/api.ts',
  apiImport: 'api',
  tag: true,
  outputFiles: {
    './src/app/rtk/generated/users.ts': {
      filterEndpoints: [/user/i],
      exportName: 'generatedUsersApi',
    },
    './src/app/rtk/generated/datasets.ts': {
      filterEndpoints: [/dataset/i, 'getS3Data'],
      exportName: 'generatedDatasetsApi',
    },
    './src/app/rtk/generated/models.ts': {
      filterEndpoints: [/model/i],
    },
    './src/app/rtk/generated/experiments.ts': {
      filterEndpoints: [/experiment/i],
    },
    './src/app/rtk/generated/units.ts': {
      filterEndpoints: [/unit/i],
    },
    './src/app/rtk/generated/events.ts': {
      filterEndpoints: [/event/i],
    },
    './src/app/rtk/generated/auth.ts': {
      filterEndpoints: [/login/i, /oauth/i],
    },
    './src/app/rtk/generated/deployments.ts': {
      filterEndpoints: [/deployment/i],
      exportName: 'generatedDeploymentsApi',
    },
  },
  hooks: { queries: true, mutations: true, lazyQueries: true },
};
export default config;
