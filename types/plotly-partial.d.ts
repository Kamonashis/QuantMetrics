// The plotly.js partial dist bundles ship without typings; they expose the
// same API as plotly.js (restricted to the bundled trace types).
declare module 'plotly.js-cartesian-dist-min' {
  import Plotly from 'plotly.js';
  export default Plotly;
}

declare module 'plotly.js-gl3d-dist-min' {
  import Plotly from 'plotly.js';
  export default Plotly;
}
