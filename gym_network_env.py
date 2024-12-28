import gymnasium as gym
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class ITNetworkEnv(gym.Env):
    def __init__(self, graph, start_node=None, end_node=None):
        super().__init__()
        self.graph = graph
        self.nodes = list(graph.nodes)

        # set the start node if provided, random otherwise
        if start_node is None:
            self.start_node = np.random.choice(self.nodes)  # Define a start node
        else:
            self.start_node = list(self.nodes)[start_node]

        # set the end node if provided, random otherwise
        if end_node is None:
            self.end_node = np.random.choice(self.nodes)  # Define an end node
            while self.end_node == self.start_node:
                self.end_node = np.random.choice(self.nodes)
        else:
            self.end_node = self.nodes[end_node]
        
        # Start at the start node
        self.current_node = self.start_node  

        # Observation space: the index of the current node
        self.observation_space = gym.spaces.Discrete(len(self.nodes))

        # Update the action space for the current node
        self._update_action_space(self.current_node)


    def _update_action_space(self, node):
        """Update the action space for a given node"""

        # Action space: the index of the target node
        self.action_space = [idx for idx in self.graph[node]]

    def reset(self):
        """Reset the environment to the start node"""

        self.current_node = self.start_node
        return self.current_node, {}

    def step(self, action):
        """Step through the environment based on a provided action"""

        target_node = list(self.graph.nodes)[action]

        if self.graph.nodes[target_node]["allowed"]:
            self.current_node = target_node
            reward = self.graph.nodes[target_node]["reward"]
        else:
            reward = -20
        
        done = target_node == self.end_node

        # update the action space
        self._update_action_space(self.current_node)

        # state, reward, done, info
        return self.current_node, reward, done, {}

    def render(self, mode="human"):
        """Render the graph"""

        self._visualize_graph()

    def _visualize_graph(self):
        """Renders the graph with matplotlib"""

        colors = []
        for node in self.graph.nodes:
            if node == self.start_node:
                colors.append("blue")  # Start node color
            elif node == self.end_node:
                colors.append("yellow")  # End node color
            elif self.graph.nodes[node]["allowed"]:
                colors.append("green")  # allowed node color
            else:
                colors.append("red")  # Not allowed node color

        pos = nx.spring_layout(self.graph)

        labels = nx.get_node_attributes(self.graph, "reward")
        labels = { idx : f"{idx}:{labels[idx]}" for idx in labels}

        nx.draw(self.graph, pos, labels=labels, with_labels=True, node_color=colors, node_size=700, font_weight="bold")
        plt.legend(handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Start Node"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow", markersize=10, label="End Node"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="Allowed"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Not Allowed")
        ])
        plt.show(block=False)

# Create a sample graph
def create_sample_graph():
    """Create a sample graph"""

    G = nx.Graph()

    # Add nodes
    G.add_node(0, allowed=True, reward=-1)
    G.add_node(1, allowed=False, reward=-20)
    G.add_node(2, allowed=True, reward=-10)
    G.add_node(3, allowed=True, reward=-2)
    G.add_node(4, allowed=False, reward=-20)
    G.add_node(5, allowed=False, reward=-20)
    G.add_node(6, allowed=True, reward=-3)
    G.add_node(7, allowed=True, reward=-2)
    G.add_node(8, allowed=True, reward=10)
    G.add_node(9, allowed=False, reward=-20)

    # Make connections
    G.add_edge(list(G.nodes)[0], list(G.nodes)[1])
    G.add_edge(list(G.nodes)[0], list(G.nodes)[2])
    G.add_edge(list(G.nodes)[0], list(G.nodes)[3])
    G.add_edge(list(G.nodes)[1], list(G.nodes)[4])
    G.add_edge(list(G.nodes)[2], list(G.nodes)[5])
    G.add_edge(list(G.nodes)[2], list(G.nodes)[8])
    G.add_edge(list(G.nodes)[3], list(G.nodes)[6])
    G.add_edge(list(G.nodes)[4], list(G.nodes)[7])
    G.add_edge(list(G.nodes)[5], list(G.nodes)[6])
    G.add_edge(list(G.nodes)[6], list(G.nodes)[7])
    G.add_edge(list(G.nodes)[7], list(G.nodes)[8])
    G.add_edge(list(G.nodes)[8], list(G.nodes)[9])

    return G

def main():
    # Instantiate and visualize the environment
    sample_graph = create_sample_graph()
    env = ITNetworkEnv(sample_graph, start_node=0, end_node=8)

    print(env.action_space)
    print(env.observation_space)
    print(f"Number of actions for {env.current_node}: {env.action_space}")
    print(f"Number of states: {env.observation_space.n}")

    env.render()
    done = False
    while not done:
        print(f"Possible actions: {env.action_space}")
        action = int(input())
        state, reward, done, info = env.step(action)
        print(f"Current state: {state}")
        print(f"Reward for this action: {reward}")
        print(f"Done: {done}")

if __name__ == "__main__":
    main()

