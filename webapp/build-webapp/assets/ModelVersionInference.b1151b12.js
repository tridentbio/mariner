import{aY as i,O as c,r as m,f,H as u,j as o,C as p,bH as I,aW as M}from"./index.5bdb0a85.js";const V=()=>o(M,{children:"Model version not found :( "}),h=()=>{const s=i("/models/:modelName/:modelVersion/inference"),{modelName:e,modelVersion:n}=(s==null?void 0:s.params)||{},[t,{isLoading:l,data:r}]=c.useLazyGetModelByIdQuery();m.exports.useEffect(()=>{const a=e&&parseInt(e);a&&t(a)},[e]);const d=n&&parseInt(n);return f(u,{children:[(!e||!n)&&o(V,{}),l&&o(p,{}),r&&d&&o(I,{model:r,modelVersionId:d})]})};export{h as default};
