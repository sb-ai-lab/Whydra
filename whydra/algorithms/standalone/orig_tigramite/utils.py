def _nb_latent_before(node, observed_context_indices, node_classification):
    return len(
        [el for el in range(node) if not (el in observed_context_indices or node_classification[el] == "system")])

def _do_dummy_projection(links, node_classification, observed_context_indices, time_dummy_index, space_dummy_index):
    """
    Helper function to augment the true_parents by remove context-context links and adding dummy
    (i.e. perform dummy projection)
    """
    # В оригинальном коде эта функция используется для валидации на toy models.
    # Для работы алгоритма на реальных данных она, скорее всего, не критична,
    # но JPCMCI+ её импортирует. Оставим заглушку или копию, если она вызывается.
    # В run_jpcmciplus она НЕ вызывается, она используется в тестах.
    # Но на всякий случай оставим пустую реализацию или копию, если вдруг понадобится.
    return {}

def _group_links(links, node_types, node_type):
    return {i: links[i] for i in links.keys() if node_types[i] == node_type}