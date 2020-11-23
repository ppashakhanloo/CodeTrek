

def __deepcopy__(self, memo):
    'Method used by copy.deepcopy().  This also uses the\n        state_pickler to work correctly.\n        '
    new = self.__class__()
    saved_state = self._saved_state
    if (len(saved_state) == 0):
        state = state_pickler.get_state(self)
        if (not is_old_pipeline()):
            try:
                st = state.children[0].children[4]
                l_pos = st.seed.widget.position
                st.seed.widget.position = [pos.item() for pos in new]
            except (IndexError, AttributeError):
                pass
        saved_state = pickle.dumps(state)
    new._saved_state = saved_state
    if new.running:
        new._load_saved_state()
    return new
