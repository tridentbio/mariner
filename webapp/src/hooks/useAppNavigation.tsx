import {
  matchPath,
  Params,
  RouteObject,
  useLocation,
  useRoutes,
} from 'react-router-dom';
import { useState, useEffect, useMemo, lazy } from 'react';

import { RootState, store } from 'app/store';
import { Model, ModelVersion } from 'app/types/domain/models';
import TrainingListing from 'features/trainings/pages/TrainingListing';
import CreateTraining from 'features/trainings/pages/CreateTraining';
import { AppLayout } from 'components/templates/AppLayout';
import ModelDetails from 'features/models/pages/ModelDetails';

const Dashboard = lazy(() => import('../features/dashboard/Dashboard'));
// const ModelDetails = import('../features/models/pages/ModelDetails');
const ModelCreate = lazy(
  () => import('../features/models/pages/ModelCreateV2')
);
const ModelVersionDetails = lazy(
  () => import('../features/models/pages/ModelVersionDetails')
);
const AuthenticationPage = lazy(
  () => import('../features/users/Authentication')
);
const DatasetListing = lazy(
  () => import('../features/datasets/pages/DatasetListing')
);
const DatasetEdit = lazy(
  () => import('../features/datasets/pages/DatasetEdit')
);
const DatasetDetails = lazy(
  () => import('../features/datasets/pages/DatasetDetails')
);
const RequireAuth = lazy(() => import('../components/templates/RequireAuth'));
const ModelVersionInference = lazy(
  () => import('../features/models/pages/ModelVersionInference')
);
const ModelListing = lazy(
  () => import('../features/models/pages/ModelListing')
);
const DeploymentsListing = lazy(
  () => import('../features/deployments/Pages/DeploymentsListing')
);
const DeploymentView = lazy(
  () => import('../features/deployments/Pages/DeploymentView')
);
const PublicDeploymentView = lazy(
  () => import('../features/deployments/Pages/DeploymentViewPublic')
);

type Breadcrumb = {
  label: string;
  url: string;
};
type RouteNode<T extends {}> = Omit<RouteObject, 'children'> & {
  breadcrumb?: Breadcrumb;
  children?: RouteNode<any>[];
  withSelector?: (
    state: RootState,
    routeParams: Params<string>
  ) => T | undefined;
  onUndefinedSelector?: () => any;
};
const findDatasetMatchingParamsId = (
  state: RootState,
  params: Params<string>
) =>
  state.datasets.datasets.find(
    (dataset) => dataset.id.toString() === params.id
  );

/**
 * path should have `:modelId`
 */
const findModelMatchingParamsId = (
  state: RootState,
  params: Params<string>
): Model | undefined =>
  state.models.models.find((model) => model.id.toString() === params.modelId);

export const getByFieldPath = <T extends { [key: string]: any }>(
  path: string,
  obj: T
): any => {
  const nodes = path.split('.');
  let value: typeof obj | undefined = obj;
  for (const node of nodes) {
    if (typeof value === 'object' && node in value) {
      value = value && node in value ? value[node] : undefined;
    } else {
      value = undefined;
    }
  }
  return value;
};

/**
 * path should have `:modelId` and `:modelVersionId`
 */
const findModelVersionParamsId = (
  state: RootState,
  params: Params<string>
): { model: Model; version: ModelVersion } | undefined => {
  for (const model of state.models.models) {
    const version = model.versions.find((version) => {
      return version.id.toString() === params.modelVersionId;
    });
    if (version) return { model, version };
  }
};

/**
 * path should have `:deploymentId`
 */
const findDeploymentMatchingParamsId = (
  state: RootState,
  params: Params<string>
) => {
  return state.deployments.deployments.find(
    (deployment) => deployment.id.toString() === params.deploymentId
  );
};

const navigationTree: RouteNode<any>[] = [
  {
    path: '/login/*',
    element: <AuthenticationPage />,
  },
  {
    path: '/public-model/*',
    children: [
      {
        path: ':token1/:token2/:token3/*',
        element: <PublicDeploymentView />,
      },
    ],
  },
  {
    breadcrumb: {
      label: 'Home',
      url: '/',
    },
    element: (
      <RequireAuth>
        <AppLayout />
      </RequireAuth>
    ),
    children: [
      {
        path: '',
        element: <Dashboard />,
      },
      {
        path: '/datasets/',
        breadcrumb: {
          label: 'Datasets',
          url: '/datasets',
        },
        children: [
          {
            path: '',
            element: <DatasetListing />,
          },
          {
            path: ':id/',
            withSelector: findDatasetMatchingParamsId,
            breadcrumb: {
              label: ':name',
              url: '/datasets/:id',
            },
            children: [
              {
                path: '',
                element: <DatasetDetails />,
              },
              {
                path: 'edit',
                element: <DatasetEdit />,
                withSelector: findDatasetMatchingParamsId,
                breadcrumb: {
                  label: 'Edit :name',
                  url: '/datasets/:id',
                },
              },
            ],
          },
        ],
      },
      {
        path: '/trainings/',
        breadcrumb: {
          label: 'Trainings',
          url: '/trainings',
        },
        children: [
          {
            path: '',
            element: <TrainingListing />,
          },
          {
            path: 'new/*',
            breadcrumb: {
              label: 'Create Training',
              url: '/trainings/new',
            },
            element: <CreateTraining />,
          },
          {
            path: ':id/',
            // withSelector: findDatasetMatchingParamsId,
            breadcrumb: {
              label: ':name',
              url: '/datasets/:id',
            },
            children: [
              {
                path: '',
                element: <TrainingListing />,
              },
              {
                path: 'results',
                element: <TrainingListing />,
                // withSelector: findDatasetMatchingParamsId,
                breadcrumb: {
                  label: 'Results :name',
                  url: '/datasets/:id/results',
                },
              },
            ],
          },
        ],
      },
      {
        path: '/models/',
        breadcrumb: {
          label: 'Models',
          url: '/models',
        },
        children: [
          {
            path: 'new',
            breadcrumb: {
              label: 'Create Model',
              url: '/models/new',
            },
            element: <ModelCreate />,
          },
          {
            path: '',
            element: <ModelListing />,
          },
          {
            withSelector: findModelMatchingParamsId,
            path: ':modelId/',
            breadcrumb: {
              label: ':name',
              url: '/models/:id',
            },
            children: [
              {
                path: '',
                element: <ModelDetails />,
              },
              {
                path: ':modelVersionId/',
                children: [
                  {
                    path: '',
                    element: <ModelVersionDetails />,
                  },
                  {
                    path: 'inference',
                    element: <ModelVersionInference />,
                    breadcrumb: {
                      label: ':model.name (:version.name) - Inference',
                      url: '/models/:model.id/:version.id/inferece',
                    },
                  },
                ],
                withSelector: findModelVersionParamsId,
                breadcrumb: {
                  label: ':model.name (:version.name)',
                  url: '/models/:model.id/:version.id',
                },
              },
            ],
          },
        ],
      },
      {
        withSelector: findDeploymentMatchingParamsId,
        path: '/deployments/',
        breadcrumb: {
          label: 'Deployments',
          url: '/deployments',
        },
        children: [
          {
            path: '',
            element: <DeploymentsListing />,
          },
          {
            path: ':deploymentId/',
            breadcrumb: {
              label: ':name',
              url: '/deployments/:id',
            },
            children: [
              {
                path: '',
                element: <DeploymentView />,
              },
            ],
          },
        ],
      },
    ],
  },
];
function useAppNavigation() {
  const getMatchingBranch = (routes: RouteNode<any>[], pathname: string) => {
    const getBranchRecursive = (
      nodes: RouteNode<any>[],
      pathname: string,
      parentPath: string = '',
      routePath: RouteNode<any>[] = []
    ): RouteNode<any>[] => {
      for (const node of nodes) {
        if (
          parentPath + node.path &&
          matchPath(parentPath + node.path, pathname)
        ) {
          return [...routePath, node];
        } else if (node?.children) {
          const branch = getBranchRecursive(
            node.children,
            pathname,
            parentPath + (node.path || ''),
            [...routePath, node]
          );
          if (branch.length) return branch;
        }
      }
      return [];
    };
    return getBranchRecursive(routes, pathname);
  };

  /**
   * replace all substrings matching `:<key>` (where key is runs until the
   * '/' char) by the value in the params[key]
   * Returns true if some `:<key>` substring were not replaced, false otherwise
   */
  const replace = (str: string, params: Params<string>): [string, boolean] => {
    //@ts-ignore
    const re = /:[\w\.]*/gm;
    let someMissing = false;
    let match: RegExpExecArray | null = re.exec(str);
    if (match === null) return [str, false];
    do {
      const index = match.index;
      if (index === undefined) continue;
      const key = match[0].slice(1);
      const storeValue = getByFieldPath(key, params);

      if (!storeValue) {
        someMissing = true;
        match = re.exec(str);
      } else {
        str =
          str.slice(0, index) + storeValue + str.slice(index + match[0].length);
        re.lastIndex = 0;
        match = re.exec(str);
      }
    } while (match);
    return [str, someMissing];
  };

  const replaceBreacrumbLabelField = (
    bread: Breadcrumb,
    params: Params<string>
  ): [Breadcrumb, boolean] => {
    const [label, missingInLabel] = replace(bread.label, params);
    const [url, missingInUrl] = replace(bread.url, params);
    return [
      {
        ...bread,
        label,
        url,
      },
      missingInLabel || missingInUrl,
    ];
  };

  const getBreadcrumbs = (routeNodes: RouteNode<any>[]): Breadcrumb[] => {
    const breads = routeNodes
      .filter((route) => !!route.breadcrumb)
      .map((route) => route.breadcrumb!);
    return breads;
  };
  const { pathname } = useLocation();
  const branch = useMemo(() => {
    return getMatchingBranch(navigationTree, pathname);
  }, [pathname, navigationTree]);
  const [breadcrumbs, setBreadcrumbs] = useState<Breadcrumb[]>([]);
  const [missing, setMissing] = useState(true);
  const getProcessedBreacrumbs = (
    branch: RouteNode<any>[],
    pathname: string,
    state: RootState
  ): [Breadcrumb[], boolean] => {
    const brs = getBreadcrumbs(branch);
    const selectors = branch
      .filter((node) => !!node.withSelector)
      .map((node) => node.withSelector!);
    const branchPath = branch.map((route) => route.path || '').join('');
    const params = matchPath(branchPath, pathname)?.params || {};
    const storeParams: Params<string> = selectors.reduce((acc, select) => {
      const data = select(state, params);
      return {
        ...acc,
        ...data,
      };
    }, {});
    let someMissing = false;
    const crumbs: Breadcrumb[] = [];
    brs.forEach((crumb) => {
      const [processedCrumb, missing] = replaceBreacrumbLabelField(crumb, {
        ...params,
        ...storeParams,
      });
      someMissing = someMissing || missing;
      crumbs.push(processedCrumb);
    });
    return [crumbs, someMissing];
  };
  useEffect(() => {
    const [crumbs, someMissing] = getProcessedBreacrumbs(
      branch,
      pathname,
      store.getState()
    );
    setBreadcrumbs(crumbs);
    setMissing(someMissing);
    return store.subscribe(() => {
      if (!missing) return;
      const [crumbs, someMissing] = getProcessedBreacrumbs(
        branch,
        pathname,
        store.getState()
      );
      setBreadcrumbs(crumbs);
      setMissing(someMissing);
    });
  }, [pathname]);

  const routes = useRoutes(navigationTree as RouteObject[]);
  return {
    routes,
    breadcrumbs,
  };
}

export default useAppNavigation;
