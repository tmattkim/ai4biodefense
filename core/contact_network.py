"""Multi-layer contact network construction.

Three layers:
1. Household: complete graph within each household
2. Workplace/School: random regular graphs for ages 5-64
3. Community: age-assortative random edges using POLYMOD matrix
"""

import numpy as np
import networkx as nx


def get_polymod_matrix() -> np.ndarray:
    """Return 5x5 POLYMOD contact matrix (Mossong et al. 2008).

    Rows/columns correspond to age bins: 0-4, 5-17, 18-49, 50-64, 65+.
    Values represent mean daily contacts between age groups,
    aggregated from the original 15 age bins to our 5 bins.
    """
    return np.array([
        [2.16, 1.03, 3.51, 0.75, 0.50],
        [1.03, 7.65, 3.85, 1.14, 0.42],
        [3.51, 3.85, 6.25, 1.88, 0.93],
        [0.75, 1.14, 1.88, 2.51, 0.98],
        [0.50, 0.42, 0.93, 0.98, 1.45],
    ])


def _build_household_network(agents: list) -> nx.Graph:
    """Build household contact layer: complete graph within each household."""
    G = nx.Graph()
    G.add_nodes_from(a.id for a in agents)

    # Group agents by household
    households = {}
    for a in agents:
        households.setdefault(a.household_id, []).append(a.id)

    for members in households.values():
        if len(members) > 1:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    G.add_edge(members[i], members[j], layer='household')

    return G


def _build_workplace_school_network(
    agents: list, group_size: int = 20, degree: int = 8, rng: np.random.Generator = None
) -> nx.Graph:
    """Build workplace/school contact layer.

    Ages 5-64 are assigned to groups (workplaces/schools) of ~group_size.
    Within each group, connections form a random regular-ish graph.
    Ages 0-4 and 65+ are excluded (preschool/retired).
    """
    G = nx.Graph()
    G.add_nodes_from(a.id for a in agents)

    # Filter eligible agents
    eligible = [a.id for a in agents if 5 <= a.age <= 64]
    if rng is not None:
        rng.shuffle(eligible)

    # Partition into groups
    groups = []
    for i in range(0, len(eligible), group_size):
        groups.append(eligible[i:i + group_size])

    # Within each group, create random edges
    for group in groups:
        if len(group) < 2:
            continue
        n_g = len(group)
        effective_degree = min(degree, n_g - 1)
        # Use Erdos-Renyi within group to approximate random regular
        p = effective_degree / (n_g - 1) if n_g > 1 else 0
        for i in range(n_g):
            for j in range(i + 1, n_g):
                if rng is not None and rng.random() < p:
                    G.add_edge(group[i], group[j], layer='workplace_school')

    return G


def _build_community_network(
    agents: list, contact_matrix: np.ndarray, mean_community_degree: float = 5.0,
    rng: np.random.Generator = None
) -> nx.Graph:
    """Build community contact layer with age-assortative mixing.

    Uses POLYMOD matrix to determine relative contact probabilities
    between age groups. Overall mean degree is ~mean_community_degree.
    """
    G = nx.Graph()
    n = len(agents)
    G.add_nodes_from(a.id for a in agents)

    # Group agents by age bin
    age_bin_agents = {}
    for a in agents:
        age_bin_agents.setdefault(a.age_bin, []).append(a.id)

    # Normalize contact matrix to get mixing probabilities
    total_contacts = contact_matrix.sum()
    mix_prob = contact_matrix / total_contacts

    # Calculate target number of community edges
    target_edges = int(n * mean_community_degree / 2)

    edges_added = 0
    max_attempts = target_edges * 10

    for _ in range(max_attempts):
        if edges_added >= target_edges:
            break

        # Sample an age-bin pair weighted by mixing probabilities
        flat_idx = rng.choice(mix_prob.size, p=mix_prob.ravel())
        bin_i, bin_j = divmod(flat_idx, contact_matrix.shape[1])

        agents_i = age_bin_agents.get(bin_i, [])
        agents_j = age_bin_agents.get(bin_j, [])

        if not agents_i or not agents_j:
            continue

        a_i = rng.choice(agents_i)
        a_j = rng.choice(agents_j)

        if a_i != a_j and not G.has_edge(a_i, a_j):
            G.add_edge(a_i, a_j, layer='community')
            edges_added += 1

    return G


def build_contact_network(
    agents: list,
    rng: np.random.Generator,
    contact_matrix: np.ndarray = None,
    mean_community_degree: float = 5.0,
    workplace_group_size: int = 20,
    workplace_degree: int = 8,
) -> dict:
    """Build the complete multi-layer contact network.

    Args:
        agents: List of Agent objects.
        rng: NumPy random generator.
        contact_matrix: 5x5 POLYMOD matrix (uses default if None).
        mean_community_degree: Target mean degree for community layer.
        workplace_group_size: Size of workplace/school groups.
        workplace_degree: Target degree within workplace groups.

    Returns:
        Dict with keys 'household', 'workplace_school', 'community',
        each containing a NetworkX Graph.
    """
    if contact_matrix is None:
        contact_matrix = get_polymod_matrix()

    household = _build_household_network(agents)
    workplace = _build_workplace_school_network(
        agents, workplace_group_size, workplace_degree, rng
    )
    community = _build_community_network(
        agents, contact_matrix, mean_community_degree, rng
    )

    return {
        'household': household,
        'workplace_school': workplace,
        'community': community,
    }


def get_combined_adjacency(networks: dict, agents: list) -> dict:
    """Build combined adjacency lists from all network layers.

    Returns:
        Dict mapping agent_id -> list of (neighbor_id, layer_weight) tuples.
        Layer weights: household=1.0, workplace_school=0.5, community=0.3
    """
    LAYER_WEIGHTS = {
        'household': 1.0,
        'workplace_school': 0.5,
        'community': 0.3,
    }

    adjacency = {a.id: [] for a in agents}

    for layer_name, G in networks.items():
        weight = LAYER_WEIGHTS[layer_name]
        for u, v in G.edges():
            adjacency[u].append((v, weight))
            adjacency[v].append((u, weight))

    return adjacency
