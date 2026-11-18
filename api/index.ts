import worker from '../worker.mjs';

export default {
  fetch(request: Request): Promise<Response> {
    return worker.fetch(request);
  },
};
