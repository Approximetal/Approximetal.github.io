import type { Component } from "solid-js";
import type { ProjectItem } from "@/types";

export const Project: Component<{ project: ProjectItem }> = (props) => {
  return (
    <a
      class="project-card relative block p-4 !no-underline !text-fg"
      href={props.project.link}
      title={props.project.name}
      target="_blank"
      rel="noopener noreferrer"
    >
      <div whitespace-nowrap mr-3>
        {props.project.name}
      </div>
      <div mt-1 text="sm fg-light" innerHTML={props.project.desc} />
    </a>
  );
};

export default Project;
